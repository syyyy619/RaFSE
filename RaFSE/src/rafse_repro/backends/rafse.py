from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from ..dynamic_k import (
    DynamicKThresholds,
    assign_dynamic_k,
    distribution,
)
from ..errors import ProtocolError
from ..hashing import sha256_file
from ..metrics import TOPK, evaluate_top200
from .common import (
    add_in_chunks,
    normalize_rows,
    reconstruct_many,
    require_faiss,
    search_in_batches,
    suppress_native_stderr,
    time_search,
)


@dataclass(frozen=True)
class RafseRunConfig:
    gallery_size: int
    threads: int = 16
    query_batch_size: int = 512
    add_batch_size: int = 50_000
    repeats: int = 1
    pq_seed: int = 1234
    coarse_m: int = 64
    coarse_nbits: int = 8
    residual_m: int = 64
    residual_nbits: int = 8
    candidate_bytes: int = 128

    def __post_init__(self) -> None:
        if self.gallery_size not in (100_000, 1_000_000):
            raise ProtocolError("RaFSE paper runs support only 100K and 1M")
        if self.threads != 16:
            raise ProtocolError("Paper latency protocol requires 16 Faiss threads")


def _train_pq(train: np.ndarray, m: int, nbits: int, seed: int):
    faiss = require_faiss()
    if train.shape[1] % m:
        raise ProtocolError(f"Descriptor dimension {train.shape[1]} is not divisible by m={m}")
    index = faiss.IndexPQ(train.shape[1], m, nbits, faiss.METRIC_L2)
    seed_applied = False
    if hasattr(index.pq, "cp") and hasattr(index.pq.cp, "seed"):
        index.pq.cp.seed = int(seed)
        seed_applied = True
    started = time.perf_counter()
    with suppress_native_stderr():
        index.train(np.ascontiguousarray(train, dtype=np.float32))
    return index, time.perf_counter() - started, seed_applied


def _reconstruct_range(index, start: int, count: int) -> np.ndarray:
    if hasattr(index, "reconstruct_n"):
        return np.asarray(index.reconstruct_n(start, count), dtype=np.float32)
    return reconstruct_many(index, np.arange(start, start + count, dtype=np.int64))


def _build_residual_gallery(
    coarse_index,
    residual_index,
    gallery: np.ndarray,
    batch_size: int,
) -> float:
    started = time.perf_counter()
    for begin in range(0, len(gallery), batch_size):
        end = min(begin + batch_size, len(gallery))
        normalized = normalize_rows(np.asarray(gallery[begin:end]))
        coarse = _reconstruct_range(coarse_index, begin, end - begin)
        residual_index.add(np.ascontiguousarray(normalized - coarse, dtype=np.float32))
    return time.perf_counter() - started


def _rerank_dynamic(
    queries: np.ndarray,
    coarse_rankings: np.ndarray,
    coarse_index,
    residual_index,
    assigned_k: np.ndarray,
    batch_size: int,
    timed: bool,
) -> tuple[np.ndarray | None, float]:
    reranked = None if timed else coarse_rankings.copy()
    elapsed = 0.0
    for begin in range(0, len(queries), batch_size):
        end = min(begin + batch_size, len(queries))
        batch_rankings = coarse_rankings[begin:end]
        candidate_ids = np.unique(batch_rankings.reshape(-1))
        refined = reconstruct_many(coarse_index, candidate_ids) + reconstruct_many(
            residual_index, candidate_ids
        )
        id_to_position = {int(value): pos for pos, value in enumerate(candidate_ids.tolist())}
        started = time.perf_counter()
        for local_index, query in enumerate(queries[begin:end]):
            prefix_k = int(assigned_k[begin + local_index])
            prefix = batch_rankings[local_index, :prefix_k]
            positions = np.fromiter(
                (id_to_position[int(value)] for value in prefix),
                dtype=np.int32,
                count=prefix_k,
            )
            candidates = refined[positions]
            squared_l2 = np.sum((candidates - query) ** 2, axis=1)
            order = np.argsort(squared_l2)
            if reranked is not None:
                reranked[begin + local_index, :prefix_k] = prefix[order]
        elapsed += time.perf_counter() - started
    return reranked, elapsed


def run_rafse_dynamic(
    *,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    train_features: np.ndarray,
    thresholds_path: Path,
    output_dir: Path,
    config: RafseRunConfig,
) -> dict[str, Any]:
    """Run RaFSE with the scale-specific frozen Dynamic-K thresholds."""

    faiss = require_faiss()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProtocolError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if len(query_features) != 37_855:
        raise ProtocolError(f"Strict run requires 37,855 queries; got {len(query_features)}")
    if len(gallery_features) != config.gallery_size:
        raise ProtocolError(
            f"Gallery/config mismatch: {len(gallery_features)} vs {config.gallery_size}"
        )
    if not thresholds_path.exists():
        raise ProtocolError(f"Missing frozen threshold manifest: {thresholds_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(config.threads)
    queries = normalize_rows(query_features)
    train = normalize_rows(train_features)

    stage_started = time.perf_counter()
    coarse_index, coarse_train_s, coarse_seed_applied = _train_pq(
        train, config.coarse_m, config.coarse_nbits, config.pq_seed
    )
    coarse_index.add(train)
    train_coarse = _reconstruct_range(coarse_index, 0, len(train))
    coarse_index.reset()
    coarse_add_s = add_in_chunks(coarse_index, gallery_features, config.add_batch_size)

    train_residual = np.ascontiguousarray(train - train_coarse, dtype=np.float32)
    residual_index, residual_train_s, residual_seed_applied = _train_pq(
        train_residual, config.residual_m, config.residual_nbits, config.pq_seed
    )
    residual_add_s = _build_residual_gallery(
        coarse_index, residual_index, gallery_features, config.add_batch_size
    )

    coarse_distances, coarse_rankings = search_in_batches(
        coarse_index,
        queries,
        TOPK,
        config.query_batch_size,
        keep_distances=True,
    )
    assert coarse_distances is not None
    margins = coarse_distances[:, 1] - coarse_distances[:, 0]
    thresholds = DynamicKThresholds.from_json(thresholds_path)
    if thresholds.gallery_size != config.gallery_size:
        raise ProtocolError("Threshold/gallery scale mismatch")

    assigned_k = assign_dynamic_k(margins, thresholds)
    reranked, _ = _rerank_dynamic(
        queries,
        coarse_rankings,
        coarse_index,
        residual_index,
        assigned_k,
        config.query_batch_size,
        timed=False,
    )
    assert reranked is not None
    metrics = evaluate_top200(reranked, query_labels, gallery_labels)

    coarse_timings = time_search(
        coarse_index, queries, TOPK, config.query_batch_size, config.repeats
    )
    rerank_timings = []
    for _ in range(config.repeats):
        _, elapsed = _rerank_dynamic(
            queries,
            coarse_rankings,
            coarse_index,
            residual_index,
            assigned_k,
            config.query_batch_size,
            timed=True,
        )
        rerank_timings.append(elapsed)
    retrieval_s = float((coarse_timings + np.asarray(rerank_timings)).mean())
    k_summary = distribution(assigned_k)
    avg_k = float(k_summary["average_k"])

    result = {
        "method": "RaFSE",
        "protocol": "u1652_top200_fixed_threshold_v1",
        "gallery_size": config.gallery_size,
        "query_count": len(queries),
        "thresholds": asdict(thresholds),
        "dynamic_k": k_summary,
        "storage_B_per_image": 128,
        "verification_traffic_B_per_query": avg_k * config.candidate_bytes,
        "verification_traffic_KB_per_query": avg_k * config.candidate_bytes / 1000.0,
        "latency_ms_per_query": retrieval_s / len(queries) * 1000.0,
        "latency_scope": (
            "CPU retrieval only; excludes feature extraction, loading, index construction, "
            "PQ training, calibration, and gallery package loading"
        ),
        "metrics": metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_result = {
        "method": "RaFSE",
        "gallery_size": config.gallery_size,
        "query_count": len(queries),
        "R@1": metrics["R@1"],
        "R@5": metrics["R@5"],
        "AP@200": metrics["AP@200"],
        "average_K": avg_k,
        "traffic_KB_per_query": result["verification_traffic_KB_per_query"],
        "latency_ms_per_query": result["latency_ms_per_query"],
        "tau1": thresholds.tau1,
        "tau2": thresholds.tau2,
        "tau3": thresholds.tau3,
        "protocol": result["protocol"],
    }
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_result))
        writer.writeheader()
        writer.writerow(flat_result)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "training": {
            "coarse_train_s": coarse_train_s,
            "coarse_add_s": coarse_add_s,
            "residual_train_s": residual_train_s,
            "residual_add_s": residual_add_s,
            "coarse_seed_applied": coarse_seed_applied,
            "residual_seed_applied": residual_seed_applied,
        },
        "threshold_manifest": {
            "path": str(thresholds_path),
            "sha256": sha256_file(thresholds_path),
        },
        "runtime_s": time.perf_counter() - stage_started,
        "outputs": ["result.json", "result.csv", "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
