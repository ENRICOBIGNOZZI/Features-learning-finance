from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .train import annualized_sharpe


@dataclass
class CanonicalBasis:
    inv_sqrt_second_moment: np.ndarray
    sqrt_second_moment: np.ndarray
    eigenvectors: np.ndarray
    economic_eigenvalues: np.ndarray
    factor_columns: list[str]
    allocator_columns: list[str]

    @property
    def score_transform(self) -> np.ndarray:
        return self.inv_sqrt_second_moment @ self.eigenvectors


def _symmetric_power(matrix: np.ndarray, power: float, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * np.power(values, power)) @ vectors.T

def fit_canonical_basis(frame: pd.DataFrame, ridge: float = 1e-5) -> CanonicalBasis:
    factor_columns = sorted(c for c in frame.columns if c.startswith("factor_"))
    allocator_columns = sorted(c for c in frame.columns if c.startswith("allocator_"))
    f = frame[factor_columns].to_numpy(dtype=float)
    b = frame[allocator_columns].to_numpy(dtype=float)
    k = f.shape[1]

    second = f.T @ f / max(len(f), 1)
    scale = float(np.trace(second) / max(k, 1))
    floor = max(ridge * max(scale, 1e-12), 1e-12)
    second = second + floor * np.eye(k)
    sqrt_second = _symmetric_power(second, 0.5, floor)
    inv_sqrt_second = _symmetric_power(second, -0.5, floor)

    whitened_allocator = b @ sqrt_second
    economic_operator = whitened_allocator.T @ whitened_allocator / max(len(b), 1)
    values, vectors = np.linalg.eigh(economic_operator)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    projected = whitened_allocator @ vectors
    signs = np.where(projected.mean(axis=0) < 0, -1.0, 1.0)
    vectors = vectors * signs
    return CanonicalBasis(
        inv_sqrt_second_moment=inv_sqrt_second,
        sqrt_second_moment=sqrt_second,
        eigenvectors=vectors,
        economic_eigenvalues=values,
        factor_columns=factor_columns,
        allocator_columns=allocator_columns,
    )

def apply_canonical_basis(frame: pd.DataFrame, basis: CanonicalBasis) -> pd.DataFrame:
    f = frame[basis.factor_columns].to_numpy(dtype=float)
    b = frame[basis.allocator_columns].to_numpy(dtype=float)
    g = f @ basis.inv_sqrt_second_moment @ basis.eigenvectors
    d = b @ basis.sqrt_second_moment @ basis.eigenvectors

    base_columns = ["date"]
    if "payoff_date" in frame.columns:
        base_columns.append("payoff_date")
    base_columns.append("portfolio_return")
    out = frame[base_columns].copy()
    for j in range(g.shape[1]):
        out[f"canonical_factor_{j:02d}"] = g[:, j]
        out[f"canonical_allocator_{j:02d}"] = d[:, j]
        out[f"contribution_{j:02d}"] = g[:, j] * d[:, j]
    reconstructed = np.sum(g * d, axis=1)
    out["reconstruction_error"] = reconstructed - frame["portfolio_return"].to_numpy()
    return out


def economic_dimension(values: np.ndarray) -> dict[str, float]:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if values.sum() <= 0:
        return {"entropy_rank": 0.0, "participation_ratio": 0.0}
    p = values / values.sum()
    positive = p[p > 0]
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    participation = float(1.0 / np.sum(p * p))
    return {"entropy_rank": entropy_rank, "participation_ratio": participation}


def summarize_canonical(canonical: pd.DataFrame) -> pd.DataFrame:
    factors = sorted(c for c in canonical.columns if c.startswith("canonical_factor_"))
    rows = []
    total_abs = sum(canonical[f"contribution_{j:02d}"].abs().mean() for j in range(len(factors)))
    for j, column in enumerate(factors):
        factor = canonical[column].to_numpy()
        contribution = canonical[f"contribution_{j:02d}"]
        rows.append({
            "factor": j + 1,
            "sharpe": annualized_sharpe(factor),
            "mean_monthly": float(np.mean(factor)),
            "vol_monthly": float(np.std(factor, ddof=1)),
            "mean_portfolio_contribution": float(contribution.mean()),
            "abs_contribution_share": float(contribution.abs().mean() / max(total_abs, 1e-12)),
        })
    return pd.DataFrame(rows)

def save_canonical_analysis(
    train_frame: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> CanonicalBasis:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basis = fit_canonical_basis(train_frame)
    values = basis.economic_eigenvalues
    shares = values / max(values.sum(), 1e-12)
    spectrum = pd.DataFrame({
        "factor": np.arange(1, len(values) + 1),
        "economic_eigenvalue": values,
        "share": shares,
        "cumulative_share": np.cumsum(shares),
    })
    spectrum.to_csv(output_dir / "economic_factor_spectrum.csv", index=False)

    dimensions = economic_dimension(values)
    with (output_dir / "economic_dimension.json").open("w", encoding="utf-8") as stream:
        json.dump(dimensions, stream, indent=2)

    for name, frame in split_frames.items():
        canonical = apply_canonical_basis(frame, basis)
        canonical.to_parquet(output_dir / f"canonical_{name}.parquet", index=False)
        summarize_canonical(canonical).to_csv(
            output_dir / f"canonical_{name}_summary.csv", index=False
        )
        curve = truncation_curve(canonical)
        curve.to_csv(output_dir / f"canonical_{name}_truncation.csv", index=False)
        with (output_dir / f"canonical_{name}_truncation_dimension.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(truncation_dimension(curve), stream, indent=2)
    np.savez(
        output_dir / "canonical_basis.npz",
        inv_sqrt_second_moment=basis.inv_sqrt_second_moment,
        sqrt_second_moment=basis.sqrt_second_moment,
        eigenvectors=basis.eigenvectors,
        economic_eigenvalues=basis.economic_eigenvalues,
    )
    return basis


def allocation_dimension(values: np.ndarray) -> dict[str, float]:
    """Effective rank of risk-standardized factor allocation intensity."""
    return economic_dimension(values)


def truncation_curve(canonical: pd.DataFrame) -> pd.DataFrame:
    """Performance retained by the first j canonical policy directions."""
    contribution_columns = sorted(
        c for c in canonical.columns if c.startswith("contribution_")
    )
    full = canonical["portfolio_return"].to_numpy(dtype=float)
    rows = []
    running = np.zeros(len(canonical), dtype=float)
    full_var = max(float(np.mean(full * full)), 1e-12)
    for j, column in enumerate(contribution_columns, start=1):
        running = running + canonical[column].to_numpy(dtype=float)
        residual = full - running
        corr = np.corrcoef(running, full)[0, 1] if np.std(running) > 0 else np.nan
        rows.append({
            "rank": j,
            "sharpe": annualized_sharpe(running),
            "correlation_with_full": float(corr),
            "relative_payoff_mse": float(np.mean(residual * residual) / full_var),
            "rms_payoff_share": float(np.sqrt(np.mean(running * running) / full_var)),
        })
    return pd.DataFrame(rows)


def truncation_dimension(curve: pd.DataFrame) -> dict[str, int]:
    """Smallest canonical rank meeting transparent reconstruction targets."""
    def first_rank(mask):
        block = curve.loc[mask, "rank"]
        return int(block.iloc[0]) if len(block) else int(curve["rank"].max())
    return {
        "rank_corr_95": first_rank(curve["correlation_with_full"] >= 0.95),
        "rank_corr_99": first_rank(curve["correlation_with_full"] >= 0.99),
        "rank_mse_05": first_rank(curve["relative_payoff_mse"] <= 0.05),
        "rank_mse_01": first_rank(curve["relative_payoff_mse"] <= 0.01),
    }
