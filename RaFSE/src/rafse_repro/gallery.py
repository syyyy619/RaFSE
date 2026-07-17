from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import ProtocolError
from .feature_io import FEATURE_DIM, load_evaluation_mat, load_feature_array
from .hashing import sha256_file


CANONICAL_SCALE_SCHEDULE = (951, 10_000, 100_000, 500_000, 1_000_000)
GALLERY_SEED = 42


@dataclass(frozen=True)
class GalleryArtifact:
    scale: int
    original_gallery_count: int
    distractor_count: int
    feature_file: str
    label_file: str
    feature_sha256: str
    label_sha256: str


def sampled_indices_by_scale(
    pool_size: int,
    requested_scales: Iterable[int],
    seed: int = GALLERY_SEED,
) -> dict[int, np.ndarray]:
    """Reproduce the original sequential RNG consumption for every canonical scale."""

    requested = set(int(scale) for scale in requested_scales)
    unsupported = requested - set(CANONICAL_SCALE_SCHEDULE)
    if unsupported:
        raise ProtocolError(f"Unsupported gallery scales: {sorted(unsupported)}")
    if pool_size <= 0:
        raise ProtocolError("Distractor pool is empty")
    rng = np.random.default_rng(seed)
    result: dict[int, np.ndarray] = {}
    for scale in CANONICAL_SCALE_SCHEDULE:
        count = scale - CANONICAL_SCALE_SCHEDULE[0]
        indices = (
            np.empty(0, dtype=np.int64)
            if count == 0
            else rng.integers(0, pool_size, size=count, dtype=np.int64)
        )
        if scale in requested:
            result[scale] = indices
        if requested and scale >= max(requested):
            break
    return result


def build_galleries(
    *,
    eval_mat: Path,
    train_drone_features: Path,
    train_satellite_features: Path,
    output_dir: Path,
    scales: Iterable[int],
    seed: int = GALLERY_SEED,
    batch_size: int = 50_000,
) -> dict[str, object]:
    scales = tuple(sorted(set(int(value) for value in scales)))
    if not scales:
        raise ProtocolError("At least one gallery scale is required")
    if seed != GALLERY_SEED:
        raise ProtocolError(f"Paper gallery construction requires seed={GALLERY_SEED}")
    evaluation = load_evaluation_mat(eval_mat, enforce_paper_shape=True)
    drone = load_feature_array(train_drone_features, "train_drone_features")
    satellite = load_feature_array(train_satellite_features, "train_satellite_features")
    # Apply the canonical float32 normalization after concatenating both views.
    pool = np.ascontiguousarray(np.concatenate([drone, satellite], axis=0), dtype=np.float32)
    pool_norms = np.linalg.norm(pool, axis=1, keepdims=True)
    pool = np.ascontiguousarray(pool / np.maximum(pool_norms, 1e-12), dtype=np.float32)
    sampled = sampled_indices_by_scale(len(pool), scales, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    max_existing_label = int(max(evaluation.query_labels.max(), evaluation.gallery_labels.max()))
    fake_label_start = max_existing_label + 10_000_000
    artifacts: list[GalleryArtifact] = []

    for scale in scales:
        feature_path = output_dir / f"gallery_{scale}_features.npy"
        label_path = output_dir / f"gallery_{scale}_labels.npy"
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype="float32", shape=(scale, FEATURE_DIM)
        )
        labels = np.lib.format.open_memmap(
            label_path, mode="w+", dtype=evaluation.gallery_labels.dtype, shape=(scale,)
        )
        original_count = len(evaluation.gallery_features)
        features[:original_count] = evaluation.gallery_features
        labels[:original_count] = evaluation.gallery_labels
        choices = sampled[scale]
        for offset in range(0, len(choices), batch_size):
            end = min(offset + batch_size, len(choices))
            start_row = original_count + offset
            end_row = original_count + end
            features[start_row:end_row] = pool[choices[offset:end]]
            labels[start_row:end_row] = np.arange(
                fake_label_start + start_row,
                fake_label_start + end_row,
                dtype=evaluation.gallery_labels.dtype,
            )
        features.flush()
        labels.flush()
        del features, labels
        artifacts.append(
            GalleryArtifact(
                scale=scale,
                original_gallery_count=original_count,
                distractor_count=scale - original_count,
                feature_file=feature_path.name,
                label_file=label_path.name,
                feature_sha256=sha256_file(feature_path),
                label_sha256=sha256_file(label_path),
            )
        )

    manifest = {
        "protocol": "u1652_mixed_training_pool_gallery_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "canonical_scale_schedule": list(CANONICAL_SCALE_SCHEDULE),
        "feature_dim": FEATURE_DIM,
        "pool": {
            "description": "University-1652 training drone and satellite descriptors, concatenated in that order",
            "drone_count": int(len(drone)),
            "satellite_count": int(len(satellite)),
            "total_count": int(len(pool)),
            "sampling": "with replacement",
            "drone_feature_sha256": sha256_file(train_drone_features),
            "satellite_feature_sha256": sha256_file(train_satellite_features),
        },
        "artifacts": [asdict(item) for item in artifacts],
    }
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
