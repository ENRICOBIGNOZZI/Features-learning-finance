from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import JKPUSData
from .train import _pack_panels


def symmetric_sqrt(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * np.sqrt(values)) @ vectors.T


def spectrum_from_grams(
    characteristic_gram: np.ndarray,
    allocation_gram: np.ndarray,
) -> np.ndarray:
    root = symmetric_sqrt(characteristic_gram)
    operator = root @ allocation_gram @ root
    values = np.linalg.eigvalsh(0.5 * (operator + operator.T))[::-1]
    return np.maximum(values, 0.0)


def effective_rank(values: np.ndarray) -> dict[str, float]:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    p = values / max(values.sum(), 1e-15)
    positive = p[p > 0]
    return {
        "entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "participation_ratio": float(1.0 / max(np.sum(p * p), 1e-15)),
    }


def estimate_policy_spectrum(
    model,
    data: JKPUSData,
    start: str,
    end: str,
    scaler,
    device: torch.device,
    batch_months: int = 12,
):
    """Estimate separable policy rank using raw stable characteristic maps.

    The implemented portfolio uses score_k(z)/rms_{k,t}.  We absorb the
    time-varying RMS normalization into the state coefficient, so the kernel
    remains phi(z)' c_t with a stable characteristic function phi.
    """
    if getattr(model, "conditional_scores", False):
        raise ValueError("Policy spectrum is defined for the separable factor model, not direct policy")
    panels = data.materialize(start, end, scaler)
    k = model.factors
    score_gram = np.zeros((k, k), dtype=np.float64)
    coefficient_gram = np.zeros((k, k), dtype=np.float64)
    months = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(panels), batch_months):
            chunk = panels[offset:offset + batch_months]
            x, state, r, offsets = _pack_panels(chunk, device)
            out = model.forward_ragged(x, state, r, offsets)
            h = model.encoder(x)
            base = x if model.linear_characteristics else h
            raw = model.factor_head(base).detach().cpu().numpy().astype(np.float64)
            alloc = out.allocator.detach().cpu().numpy().astype(np.float64)
            for row, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
                raw_m = raw[left:right]
                rms = np.sqrt(np.mean(raw_m * raw_m, axis=0) + 1e-6)
                score_gram += raw_m.T @ raw_m / max(right - left, 1)
                coefficient = alloc[row] / rms
                coefficient_gram += np.outer(coefficient, coefficient)
                months += 1
    score_gram /= max(months, 1)
    coefficient_gram /= max(months, 1)
    values = spectrum_from_grams(score_gram, coefficient_gram)
    return values, score_gram, coefficient_gram

def save_policy_spectrum(
    model,
    data: JKPUSData,
    start: str,
    end: str,
    scaler,
    device: torch.device,
    output_dir: str | Path,
    batch_months: int = 12,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values, score_gram, alloc_gram = estimate_policy_spectrum(
        model, data, start, end, scaler, device, batch_months=batch_months
    )
    shares = values / max(values.sum(), 1e-15)
    pd.DataFrame({
        "direction": np.arange(1, len(values) + 1),
        "policy_eigenvalue": values,
        "share": shares,
        "cumulative_share": np.cumsum(shares),
    }).to_csv(output_dir / "policy_spectrum.csv", index=False)
    dimension = effective_rank(values)
    with (output_dir / "policy_dimension.json").open("w", encoding="utf-8") as stream:
        json.dump(dimension, stream, indent=2)
    np.savez(
        output_dir / "policy_grams.npz",
        characteristic_gram=score_gram,
        coefficient_gram=alloc_gram,
        eigenvalues=values,
    )
    return dimension
