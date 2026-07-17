import numpy as np

from rafse_repro.gallery import sampled_indices_by_scale


def test_direct_scale_generation_preserves_original_sequential_rng_schedule():
    samples = sampled_indices_by_scale(38_555, [100_000, 1_000_000], seed=42)
    probes_100k = [0, 1, 2, 99, 1000, len(samples[100_000]) - 1]
    probes_1m = [0, 1, 2, 99, 1000, len(samples[1_000_000]) - 1]
    assert [int(samples[100_000][i]) for i in probes_100k] == [28274, 1174, 21509, 16849, 24482, 6186]
    assert [int(samples[1_000_000][i]) for i in probes_1m] == [36127, 4358, 17676, 1214, 20850, 14845]

