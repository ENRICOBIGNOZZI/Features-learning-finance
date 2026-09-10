import numpy as np

from neural_factors.inference import (
    circular_block_indices,
    paired_sharpe_difference_interval,
    sharpe_confidence_interval,
)


def test_block_bootstrap_is_reproducible_and_paired():
    indices_a = circular_block_indices(120, 12, 20, seed=5)
    indices_b = circular_block_indices(120, 12, 20, seed=5)
    assert np.array_equal(indices_a, indices_b)
    assert indices_a.shape == (20, 120)

    rng = np.random.default_rng(9)
    base = rng.normal(0.002, 0.04, 120)
    better = base + 0.004
    ci = paired_sharpe_difference_interval(
        better, base, block_length=12, replications=200, seed=3
    )
    assert ci["estimate"] > 0
    single = sharpe_confidence_interval(better, replications=200, seed=3)
    assert single["ci_low"] < single["ci_high"]
