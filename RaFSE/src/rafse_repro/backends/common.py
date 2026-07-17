from __future__ import annotations

import contextlib
import os
import time
from typing import Iterator

import numpy as np

from ..errors import ProtocolError


def require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError(
            "Faiss is required for this backend. Install the pinned faiss extra."
        ) from exc
    return faiss


@contextlib.contextmanager
def suppress_native_stderr() -> Iterator[None]:
    saved_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ProtocolError("Cannot normalize zero-length descriptor")
    return np.ascontiguousarray(array / norms)


def add_in_chunks(index, gallery: np.ndarray, batch_size: int) -> float:
    started = time.perf_counter()
    for begin in range(0, len(gallery), batch_size):
        end = min(begin + batch_size, len(gallery))
        index.add(normalize_rows(np.asarray(gallery[begin:end])))
    return time.perf_counter() - started


def search_in_batches(
    index,
    queries: np.ndarray,
    topk: int,
    batch_size: int,
    keep_distances: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    rankings = np.empty((len(queries), topk), dtype=np.int32)
    distances = np.empty((len(queries), topk), dtype=np.float32) if keep_distances else None
    for begin in range(0, len(queries), batch_size):
        end = min(begin + batch_size, len(queries))
        batch_distances, batch_rankings = index.search(
            np.ascontiguousarray(queries[begin:end], dtype=np.float32), topk
        )
        rankings[begin:end] = batch_rankings.astype(np.int32)
        if distances is not None:
            distances[begin:end] = batch_distances.astype(np.float32)
    return distances, rankings


def time_search(index, queries: np.ndarray, topk: int, batch_size: int, repeats: int) -> np.ndarray:
    if repeats <= 0:
        raise ProtocolError("repeats must be positive")
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        for begin in range(0, len(queries), batch_size):
            end = min(begin + batch_size, len(queries))
            index.search(np.ascontiguousarray(queries[begin:end], dtype=np.float32), topk)
        timings.append(time.perf_counter() - started)
    return np.asarray(timings, dtype=np.float64)


def reconstruct_many(index, ids: np.ndarray) -> np.ndarray:
    contiguous_ids = np.ascontiguousarray(ids, dtype=np.int64)
    if hasattr(index, "reconstruct_batch"):
        return np.asarray(index.reconstruct_batch(contiguous_ids), dtype=np.float32)
    vectors = np.empty((len(ids), index.d), dtype=np.float32)
    for position, gallery_id in enumerate(contiguous_ids.tolist()):
        vectors[position] = index.reconstruct(int(gallery_id))
    return vectors
