import numpy as np
import pytest

from rafse_repro.dynamic_k import DynamicKThresholds, assign_dynamic_k, calibration_indices


def thresholds():
    return DynamicKThresholds(100_000, tau1=0.1, tau2=0.2, tau3=0.3)


def test_paper_threshold_boundaries_use_strict_less_than():
    margins = np.array([0.31, 0.30, 0.25, 0.20, 0.15, 0.10, 0.09])
    assert assign_dynamic_k(margins, thresholds()).tolist() == [20, 20, 50, 50, 100, 100, 200]


def test_assignment_is_invariant_to_batch_order():
    rng = np.random.default_rng(7)
    margins = rng.random(100)
    expected = assign_dynamic_k(margins, thresholds())
    order = rng.permutation(len(margins))
    shuffled = assign_dynamic_k(margins[order], thresholds())
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    np.testing.assert_array_equal(expected, restored)


def test_calibration_split_matches_frozen_count_and_prefix():
    calibration, evaluation = calibration_indices(37_855)
    assert len(calibration) == 11_356
    assert len(evaluation) == 26_499
    assert set(calibration).isdisjoint(evaluation)
    assert np.array_equal(np.sort(np.r_[calibration, evaluation]), np.arange(37_855))

