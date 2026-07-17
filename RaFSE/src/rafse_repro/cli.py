from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .backends.rafse import RafseRunConfig, run_rafse_dynamic
from .errors import ProtocolError
from .feature_io import load_evaluation_mat, load_feature_array, load_scaled_gallery
from .gallery import build_galleries
from .metrics import evaluate_top200


def _add_run_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-mat", type=Path, required=True)
    parser.add_argument("--gallery-features", type=Path, required=True)
    parser.add_argument("--gallery-labels", type=Path, required=True)
    parser.add_argument("--train-drone", type=Path, required=True)
    parser.add_argument("--train-satellite", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rafse",
        description="RaFSE utilities for University-1652 retrieval.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gallery = commands.add_parser("build-gallery", help="Build University-1652 galleries")
    gallery.add_argument("--eval-mat", type=Path, required=True)
    gallery.add_argument("--train-drone", type=Path, required=True)
    gallery.add_argument("--train-satellite", type=Path, required=True)
    gallery.add_argument("--output-dir", type=Path, required=True)
    gallery.add_argument(
        "--scales",
        type=int,
        nargs="+",
        choices=(951, 10_000, 100_000, 500_000, 1_000_000),
        required=True,
    )
    gallery.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="Evaluate top-200 rankings")
    evaluate.add_argument("--rankings", type=Path, required=True)
    evaluate.add_argument("--eval-mat", type=Path, required=True)
    evaluate.add_argument("--gallery-labels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    rafse = commands.add_parser("run", help="Run RaFSE")
    _add_run_inputs(rafse)
    rafse.add_argument("--gallery-size", type=int, choices=(100_000, 1_000_000), required=True)
    return parser


def _load_run_inputs(args: argparse.Namespace):
    evaluation = load_evaluation_mat(args.eval_mat, enforce_paper_shape=True)
    gallery_features, gallery_labels = load_scaled_gallery(
        args.gallery_features, args.gallery_labels
    )
    drone = load_feature_array(args.train_drone, "train_drone_features")
    satellite = load_feature_array(args.train_satellite, "train_satellite_features")
    train = np.ascontiguousarray(np.concatenate([drone, satellite], axis=0), dtype=np.float32)
    return evaluation, gallery_features, gallery_labels, train


def _write_json(payload: dict, path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    if path.exists():
        raise ProtocolError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-gallery":
            manifest = build_galleries(
                eval_mat=args.eval_mat,
                train_drone_features=args.train_drone,
                train_satellite_features=args.train_satellite,
                output_dir=args.output_dir,
                scales=args.scales,
                seed=args.seed,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        if args.command == "evaluate":
            evaluation = load_evaluation_mat(args.eval_mat, enforce_paper_shape=False)
            rankings = np.load(args.rankings, mmap_mode="r")
            gallery_labels = np.load(args.gallery_labels, mmap_mode="r")
            _write_json(
                evaluate_top200(rankings, evaluation.query_labels, gallery_labels),
                args.output,
            )
            return
        if args.command == "run":
            evaluation, gallery_features, gallery_labels, train = _load_run_inputs(args)
            run_rafse_dynamic(
                query_features=evaluation.query_features,
                query_labels=evaluation.query_labels,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                train_features=train,
                thresholds_path=args.thresholds,
                output_dir=args.output_dir,
                config=RafseRunConfig(
                    gallery_size=args.gallery_size,
                    threads=args.threads,
                    query_batch_size=args.batch_size,
                    repeats=args.repeats,
                ),
            )
            return
        raise ProtocolError(f"Unknown command: {args.command}")
    except ProtocolError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
