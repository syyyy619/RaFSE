import numpy as np
import pytest

from rafse_repro.errors import ProtocolError
from rafse_repro.metrics import evaluate_top200


def test_exact_top200_metrics_and_rank_201_zero():
    gallery_labels = np.arange(250, dtype=np.int64)
    query_labels = np.array([1, 2, 3, 4], dtype=np.int64)
    rankings = np.tile(np.arange(200, dtype=np.int32), (4, 1))
    rankings[0] = np.r_[1, np.delete(np.arange(200), 1)]
    rankings[1] = np.r_[0, 1, 5, 6, 2, np.delete(np.arange(200), [0, 1, 2, 5, 6])]
    rankings[2] = np.r_[np.delete(np.arange(200), 3), 3]
    rankings[3] = np.arange(5, 205, dtype=np.int32)  # label 4 is outside top 200

    result = evaluate_top200(rankings, query_labels, gallery_labels)
    assert result["R@1"] == pytest.approx(25.0)
    assert result["R@5"] == pytest.approx(50.0)
    assert result["R@200"] == pytest.approx(75.0)
    assert result["AP@200"] == pytest.approx((1.0 + 0.1 + 0.0025) / 4 * 100)
    assert result["queries_without_positive_in_top200"] == 1


def test_junk_is_removed_before_rank_metrics():
    gallery_labels = np.arange(201, dtype=np.int64)
    gallery_labels[0] = -1
    gallery_labels[1] = 7
    gallery_labels[7] = 999
    query_labels = np.array([7])
    rankings = np.arange(200, dtype=np.int32)[None, :]
    result = evaluate_top200(rankings, query_labels, gallery_labels)
    assert result["R@1"] == 100.0
    assert result["AP@200"] == 100.0


def test_depth_other_than_200_is_rejected():
    with pytest.raises(ProtocolError):
        evaluate_top200(np.zeros((1, 1000), dtype=np.int32), np.array([1]), np.array([1]))
