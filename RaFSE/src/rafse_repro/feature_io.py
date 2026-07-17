from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from .errors import ProtocolError


FEATURE_DIM = 768
EXPECTED_QUERY_COUNT = 37_855
EXPECTED_ORIGINAL_GALLERY_COUNT = 951


@dataclass(frozen=True)
class EvaluationFeatures:
    query_features: np.ndarray
    gallery_features: np.ndarray
    query_labels: np.ndarray
    gallery_labels: np.ndarray


def _as_f32_matrix(value: Any, name: str, expected_dim: int = FEATURE_DIM) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ProtocolError(f"{name} must have shape (N, {expected_dim}); got {array.shape}")
    if not np.isfinite(array).all():
        raise ProtocolError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _as_labels(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(array == np.floor(array)):
            raise ProtocolError(f"{name} must contain integer labels")
        array = array.astype(np.int64)
    return array


def validate_l2_normalized(features: np.ndarray, name: str, atol: float = 2e-4) -> None:
    norms = np.linalg.norm(features, axis=1)
    bad = np.flatnonzero(~np.isclose(norms, 1.0, atol=atol, rtol=0.0))
    if bad.size:
        sample = bad[:5].tolist()
        raise ProtocolError(
            f"{name} is not L2-normalized within atol={atol}; bad rows include {sample}"
        )


def load_evaluation_mat(path: Path, enforce_paper_shape: bool = True) -> EvaluationFeatures:
    required = ("query_features", "gallery_features", "query_labels", "gallery_labels")
    data = sio.loadmat(path, variable_names=list(required))
    missing = [key for key in required if key not in data]
    if missing:
        raise ProtocolError(f"Missing variables in {path}: {', '.join(missing)}")

    query_features = _as_f32_matrix(data["query_features"], "query_features")
    gallery_features = _as_f32_matrix(data["gallery_features"], "gallery_features")
    query_labels = _as_labels(data["query_labels"], "query_labels")
    gallery_labels = _as_labels(data["gallery_labels"], "gallery_labels")
    if len(query_features) != len(query_labels):
        raise ProtocolError("query feature/label length mismatch")
    if len(gallery_features) != len(gallery_labels):
        raise ProtocolError("gallery feature/label length mismatch")
    if enforce_paper_shape:
        if len(query_features) != EXPECTED_QUERY_COUNT:
            raise ProtocolError(
                f"Paper protocol requires {EXPECTED_QUERY_COUNT} queries; got {len(query_features)}"
            )
        if len(gallery_features) != EXPECTED_ORIGINAL_GALLERY_COUNT:
            raise ProtocolError(
                "Paper protocol requires an original gallery of "
                f"{EXPECTED_ORIGINAL_GALLERY_COUNT}; got {len(gallery_features)}"
            )
    validate_l2_normalized(query_features, "query_features")
    validate_l2_normalized(gallery_features, "gallery_features")
    return EvaluationFeatures(query_features, gallery_features, query_labels, gallery_labels)


def load_feature_array(path: Path, name: str, mmap: bool = False) -> np.ndarray:
    raw = np.load(path, mmap_mode="r" if mmap else None)
    features = _as_f32_matrix(raw, name)
    validate_l2_normalized(features, name)
    return features


def load_scaled_gallery(feature_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(feature_path, mmap_mode="r")
    labels = _as_labels(np.load(label_path, mmap_mode="r"), "gallery_labels")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ProtocolError(f"gallery features must have shape (N, {FEATURE_DIM}); got {features.shape}")
    if features.dtype != np.float32:
        raise ProtocolError(f"gallery features must be float32; got {features.dtype}")
    if len(features) != len(labels):
        raise ProtocolError("gallery feature/label length mismatch")
    return features, labels

