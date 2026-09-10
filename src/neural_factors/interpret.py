from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from .data import JKPUSData
from .model import NeuralFactorModel


def load_model(result_dir: str | Path, data: JKPUSData, device="cpu"):
    result_dir = Path(result_dir)
    config = json.load((result_dir / "config.json").open())
    train = config["train"]
    model = NeuralFactorModel(
        input_dim=data.input_dim,
        state_dim=len(data.state_columns),
        hidden=train["hidden"],
        factors=train["factors"],
        context_heads=train["context_heads"],
        use_context=config["use_context"],
        use_state=config["use_state"],
        static_allocator=config.get("static_allocator", False),
        linear_characteristics=config.get("linear_characteristics", False),
        conditional_scores=config.get("conditional_scores", False),
    ).to(device)
    state = torch.load(result_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    scaler = (
        np.asarray(config["state_mean"], dtype=np.float32),
        np.asarray(config["state_std"], dtype=np.float32),
    )
    basis = np.load(result_dir / "factors" / "canonical_basis.npz")
    transform = basis["inv_sqrt_second_moment"] @ basis["eigenvectors"]
    return model, scaler, transform

def canonical_scores(
    model: NeuralFactorModel,
    x: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    h = model.encoder(x)
    raw = model.factor_head(x if model.linear_characteristics else h)
    raw = raw / torch.sqrt(raw.square().mean(dim=0, keepdim=True) + 1e-6)
    return raw @ transform


def gradient_importance(
    model: NeuralFactorModel,
    data: JKPUSData,
    scaler,
    transform: np.ndarray,
    start: str,
    end: str,
    n_factors: int = 3,
    max_months: int = 24,
    stocks_per_month: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    transform_t = torch.tensor(transform[:, :n_factors], dtype=torch.float32)
    accum = np.zeros((n_factors, data.input_dim), dtype=float)
    counts = np.zeros(n_factors, dtype=float)
    used_months = 0

    for year in data.years(start, end):
        for panel in data.iter_year(year, start, end, scaler):
            if used_months >= max_months:
                break
            n = min(stocks_per_month, len(panel.x))
            idx = rng.choice(len(panel.x), size=n, replace=False)
            x = torch.tensor(panel.x[idx], dtype=torch.float32, requires_grad=True)
            scores = canonical_scores(model, x, transform_t)
            for j in range(n_factors):
                grad = torch.autograd.grad(
                    scores[:, j].sum(), x, retain_graph=j < n_factors - 1
                )[0]
                accum[j] += grad.detach().abs().sum(dim=0).numpy()
                counts[j] += n
            used_months += 1
        if used_months >= max_months:
            break

    rows = []
    for j in range(n_factors):
        importance = accum[j] / max(counts[j], 1.0)
        scale = max(importance.sum(), 1e-12)
        for characteristic, value in zip(data.characteristics, importance):
            rows.append({
                "factor": j + 1,
                "characteristic": characteristic,
                "importance": float(value),
                "share": float(value / scale),
            })
    return pd.DataFrame(rows).sort_values(
        ["factor", "share"], ascending=[True, False]
    ).reset_index(drop=True)


def collect_stock_sample(
    data: JKPUSData,
    scaler,
    start: str,
    end: str,
    max_rows: int = 6000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    per_month = max(64, max_rows // 24)
    for year in data.years(start, end):
        for panel in data.iter_year(year, start, end, scaler):
            n = min(per_month, len(panel.x))
            idx = rng.choice(len(panel.x), size=n, replace=False)
            chunks.append(panel.x[idx])
            if sum(len(chunk) for chunk in chunks) >= max_rows:
                return np.concatenate(chunks, axis=0)[:max_rows]
    return np.concatenate(chunks, axis=0)[:max_rows]


def permutation_interactions(
    model: NeuralFactorModel,
    x: np.ndarray,
    transform: np.ndarray,
    characteristics: list[str],
    top_characteristics: list[str],
    factor: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = torch.tensor(x, dtype=torch.float32)
    transform_t = torch.tensor(transform, dtype=torch.float32)

    def score(values: np.ndarray | torch.Tensor) -> np.ndarray:
        tensor = values if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float32)
        with torch.no_grad():
            return canonical_scores(model, tensor, transform_t)[:, factor].numpy()

    base = score(x0)
    scale = max(float(np.std(base, ddof=1)), 1e-12)
    indices = {name: characteristics.index(name) for name in top_characteristics}
    permutations = {name: rng.permutation(len(x)) for name in top_characteristics}
    single = {}
    for name in top_characteristics:
        xa = x.copy()
        col = indices[name]
        xa[:, col] = x[permutations[name], col]
        single[name] = score(xa)

    rows = []
    for pos, first in enumerate(top_characteristics):
        a = indices[first]
        for second in top_characteristics[pos + 1:]:
            b = indices[second]
            xab = x.copy()
            xab[:, a] = x[permutations[first], a]
            xab[:, b] = x[permutations[second], b]
            both = score(xab)
            contrast = base - single[first] - single[second] + both
            rows.append({
                "factor": factor + 1,
                "characteristic_1": first,
                "characteristic_2": second,
                "interaction_rms": float(np.sqrt(np.mean(contrast ** 2))),
                "normalized_interaction": float(np.sqrt(np.mean(contrast ** 2)) / scale),
            })
    return pd.DataFrame(rows).sort_values(
        "normalized_interaction", ascending=False
    ).reset_index(drop=True)


def factor_profile(
    model: NeuralFactorModel,
    data: JKPUSData,
    scaler,
    transform: np.ndarray,
    start: str,
    end: str,
    factor: int = 0,
    quantile: float = 0.1,
) -> pd.DataFrame:
    transform_t = torch.tensor(transform, dtype=torch.float32)
    differences = []
    for year in data.years(start, end):
        for panel in data.iter_year(year, start, end, scaler):
            x = torch.tensor(panel.x, dtype=torch.float32)
            with torch.no_grad():
                score = canonical_scores(model, x, transform_t)[:, factor].numpy()
            n_tail = max(1, int(len(score) * quantile))
            order = np.argsort(score)
            low = panel.x[order[:n_tail]].mean(axis=0)
            high = panel.x[order[-n_tail:]].mean(axis=0)
            differences.append(high - low)
    average = np.mean(differences, axis=0)
    out = pd.DataFrame({
        "factor": factor + 1,
        "characteristic": data.characteristics,
        "top_minus_bottom": average,
        "abs_top_minus_bottom": np.abs(average),
    })
    return out.sort_values("abs_top_minus_bottom", ascending=False).reset_index(drop=True)


def dominant_factor_profile(*args, **kwargs) -> pd.DataFrame:
    """Backward-compatible alias for canonical factor 1."""
    kwargs.setdefault("factor", 0)
    return factor_profile(*args, **kwargs)


def allocator_state_links(
    data: JKPUSData,
    result_dir: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    result_dir = Path(result_dir)
    canonical = pd.read_parquet(result_dir / "factors" / f"canonical_{split}.parquet")
    columns = sorted(c for c in canonical.columns if c.startswith("canonical_allocator_"))
    merged = canonical.merge(
        data.monthly_state[data.state_columns], left_on="date", right_index=True, how="left"
    )
    rows = []
    for j, allocator in enumerate(columns):
        for state in data.state_columns:
            corr = merged[allocator].corr(merged[state])
            rows.append({
                "factor": j + 1,
                "state": state,
                "correlation": float(corr) if np.isfinite(corr) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(
        ["factor", "correlation"], ascending=[True, False]
    ).reset_index(drop=True)


def panelwise_permutation_interactions(
    model: NeuralFactorModel,
    data: JKPUSData,
    scaler,
    transform: np.ndarray,
    start: str,
    end: str,
    top_characteristics: list[str],
    factor: int = 0,
    max_months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Interaction diagnostic preserving each month's cross-sectional context."""
    rng = np.random.default_rng(seed)
    transform_t = torch.tensor(transform, dtype=torch.float32)
    indices = {name: data.characteristics.index(name) for name in top_characteristics}
    pairs = [(a, b) for i, a in enumerate(top_characteristics) for b in top_characteristics[i + 1:]]
    squared = {pair: [] for pair in pairs}
    used = 0
    for year in data.years(start, end):
        for panel in data.iter_year(year, start, end, scaler):
            if used >= max_months:
                break
            x = panel.x.astype(np.float32, copy=True)
            x0 = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                base = canonical_scores(model, x0, transform_t)[:, factor].numpy()
            scale = max(float(np.std(base, ddof=1)), 1e-12)
            permutations = {name: rng.permutation(len(x)) for name in top_characteristics}
            single = {}
            for name in top_characteristics:
                perturbed = x.copy()
                col = indices[name]
                perturbed[:, col] = x[permutations[name], col]
                with torch.no_grad():
                    single[name] = canonical_scores(
                        model, torch.tensor(perturbed, dtype=torch.float32), transform_t
                    )[:, factor].numpy()
            for first, second in pairs:
                perturbed = x.copy()
                a, b = indices[first], indices[second]
                perturbed[:, a] = x[permutations[first], a]
                perturbed[:, b] = x[permutations[second], b]
                with torch.no_grad():
                    both = canonical_scores(
                        model, torch.tensor(perturbed, dtype=torch.float32), transform_t
                    )[:, factor].numpy()
                contrast = base - single[first] - single[second] + both
                squared[(first, second)].append(float(np.mean(contrast ** 2) / (scale ** 2)))
            used += 1
        if used >= max_months:
            break

    rows = []
    for (first, second), values in squared.items():
        rows.append({
            "factor": factor + 1,
            "characteristic_1": first,
            "characteristic_2": second,
            "normalized_interaction": float(np.sqrt(np.mean(values))) if values else np.nan,
            "months": len(values),
        })
    return pd.DataFrame(rows).sort_values("normalized_interaction", ascending=False).reset_index(drop=True)
