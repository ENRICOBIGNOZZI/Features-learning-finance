from __future__ import annotations

import numpy as np

from .train import annualized_sharpe


def circular_block_indices(
    n: int,
    block_length: int,
    replications: int,
    seed: int = 12345,
) -> np.ndarray:
    if n <= 0 or block_length <= 0 or replications <= 0:
        raise ValueError("n, block_length, and replications must be positive")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(replications, blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % n
    return indices.reshape(replications, -1)[:, :n]


def bootstrap_sharpe(
    returns,
    block_length: int = 12,
    replications: int = 5000,
    seed: int = 12345,
):
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    indices = circular_block_indices(len(values), block_length, replications, seed)
    samples = values[indices]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    sharpe = np.sqrt(12.0) * means / np.where(stds > 0, stds, np.nan)
    return sharpe


def sharpe_confidence_interval(
    returns,
    block_length: int = 12,
    replications: int = 5000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    estimate = annualized_sharpe(values)
    boot = bootstrap_sharpe(values, block_length, replications, seed)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(boot, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "block_length": int(block_length),
        "replications": int(replications),
    }


def paired_sharpe_difference_interval(
    first,
    second,
    block_length: int = 12,
    replications: int = 5000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    first, second = first[mask], second[mask]
    indices = circular_block_indices(len(first), block_length, replications, seed)
    a = first[indices]
    b = second[indices]
    sr_a = np.sqrt(12.0) * a.mean(axis=1) / a.std(axis=1, ddof=1)
    sr_b = np.sqrt(12.0) * b.mean(axis=1) / b.std(axis=1, ddof=1)
    differences = sr_a - sr_b
    alpha = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(differences, [alpha, 1.0 - alpha])
    estimate = annualized_sharpe(first) - annualized_sharpe(second)
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_positive": float(np.nanmean(differences > 0.0)),
        "block_length": int(block_length),
        "replications": int(replications),
    }
