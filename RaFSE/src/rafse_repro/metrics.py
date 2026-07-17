from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .errors import ProtocolError


TOPK = 200


def _valid_query_mask(query_labels: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    non_junk = gallery_labels[gallery_labels != -1]
    available = set(np.asarray(non_junk).tolist())
    return np.asarray(
        [(int(label) != -1) and (int(label) in available) for label in query_labels],
        dtype=bool,
    )


def _trapezoidal_ap_at_200(
    ranked_labels: np.ndarray,
    query_label: int,
    total_positives: int,
) -> float:
    hit_ranks = np.flatnonzero(ranked_labels == query_label)
    ap = 0.0
    for hit_index, zero_based_rank in enumerate(hit_ranks.tolist()):
        d_recall = 1.0 / total_positives
        precision = (hit_index + 1) / (zero_based_rank + 1)
        old_precision = hit_index / zero_based_rank if zero_based_rank else 1.0
        ap += d_recall * (old_precision + precision) / 2.0
    return ap


def evaluate_top200(
    rankings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the exact University-1652 paper protocol.

    Rankings must contain exactly 200 gallery IDs per query. Junk gallery labels
    are removed before metric computation. A positive outside the returned 200
    contributes zero, and AP uses the University-1652 trapezoidal convention
    normalized by the total number of non-junk positives in the full gallery.
    """

    rankings = np.asarray(rankings)
    query_labels = np.asarray(query_labels).reshape(-1)
    gallery_labels = np.asarray(gallery_labels).reshape(-1)
    if rankings.ndim != 2 or rankings.shape[1] != TOPK:
        raise ProtocolError(f"Strict protocol requires rankings shaped (N, {TOPK}); got {rankings.shape}")
    if rankings.shape[0] != len(query_labels):
        raise ProtocolError("rankings/query label length mismatch")
    if not np.issubdtype(rankings.dtype, np.integer):
        raise ProtocolError("rankings must contain integer gallery IDs")

    valid_mask = _valid_query_mask(query_labels, gallery_labels)
    valid_queries = int(valid_mask.sum())
    if valid_queries == 0:
        raise ProtocolError("No valid queries have a positive in the non-junk gallery")

    positive_counts = Counter(int(x) for x in gallery_labels[gallery_labels != -1].tolist())
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_200 = 0
    ap_sum = 0.0

    for query_index in np.flatnonzero(valid_mask).tolist():
        ids = rankings[query_index]
        ids = ids[(ids >= 0) & (ids < len(gallery_labels))]
        labels = gallery_labels[ids]
        labels = labels[labels != -1][:TOPK]
        query_label = int(query_labels[query_index])
        matches = labels == query_label
        hits_at_1 += int(np.any(matches[:1]))
        hits_at_5 += int(np.any(matches[:5]))
        hits_at_200 += int(np.any(matches))
        ap_sum += _trapezoidal_ap_at_200(
            labels,
            query_label,
            total_positives=positive_counts[query_label],
        )

    scale = 100.0 / valid_queries
    return {
        "protocol": "u1652_top200_v1",
        "query_count": int(len(query_labels)),
        "valid_queries": valid_queries,
        "skipped_queries": int(len(query_labels) - valid_queries),
        "queries_without_positive_in_top200": int(valid_queries - hits_at_200),
        "R@1": hits_at_1 * scale,
        "R@5": hits_at_5 * scale,
        "R@200": hits_at_200 * scale,
        "AP@200": ap_sum * scale,
    }

