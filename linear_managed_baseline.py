from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from neural_factors.data import JKPUSData
from neural_factors.train import SplitConfig, annualized_sharpe


def parse_args():
    p = argparse.ArgumentParser(description="Ridge tangency baseline on linear JKP managed portfolios")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="results/linear_managed_baseline")
    p.add_argument("--preset", choices=["paper", "pilot"], default="paper")
    return p.parse_args()


def factor_panel(data: JKPUSData, start: str, end: str, scaler) -> pd.DataFrame:
    rows = []
    names = ["intercept", *data.characteristics]
    for panel in data.materialize(start, end, scaler):
        x = panel.x.astype(np.float32, copy=False)
        r = panel.r.astype(np.float32, copy=False)
        managed = x.T @ r / max(len(r), 1)
        f = np.r_[r.mean(), managed]
        row = {"date": panel.date, "payoff_date": panel.date + pd.offsets.MonthEnd(1)}
        row.update({name: float(value) for name, value in zip(names, f)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def ridge_weights(frame: pd.DataFrame, columns: list[str], ridge: float) -> np.ndarray:
    x = frame[columns].to_numpy(dtype=float)
    mu = x.mean(axis=0)
    second = x.T @ x / len(x)
    scale = max(float(np.trace(second) / len(columns)), 1e-10)
    return np.linalg.solve(second + ridge * scale * np.eye(len(columns)), mu)


def portfolio(frame: pd.DataFrame, columns: list[str], weights: np.ndarray) -> np.ndarray:
    return frame[columns].to_numpy(dtype=float) @ weights


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    split = SplitConfig() if args.preset == "paper" else SplitConfig(
        train_start="2000-01-31", train_end="2014-11-30",
        val_start="2014-12-31", val_end="2019-11-30",
        test_start="2019-12-31", test_end="2024-11-30",
    )
    scaler = data.fit_state_scaler(split.train_start, split.train_end)
    train = factor_panel(data, split.train_start, split.train_end, scaler)
    val = factor_panel(data, split.val_start, split.val_end, scaler)
    test = factor_panel(data, split.test_start, split.test_end, scaler)
    columns = ["intercept", *data.characteristics]
    grid = np.logspace(-6, 4, 41)
    rows = []
    for ridge in grid:
        w = ridge_weights(train, columns, float(ridge))
        rows.append({
            "ridge": float(ridge),
            "train_sharpe": annualized_sharpe(portfolio(train, columns, w)),
            "val_sharpe": annualized_sharpe(portfolio(val, columns, w)),
        })
    diagnostics = pd.DataFrame(rows)
    best = diagnostics.loc[diagnostics["val_sharpe"].idxmax()]
    weights = ridge_weights(train, columns, float(best["ridge"]))
    test_return = portfolio(test, columns, weights)
    result = {
        "selected_ridge": float(best["ridge"]),
        "validation_sharpe": float(best["val_sharpe"]),
        "test_sharpe": annualized_sharpe(test_return),
        "n_factors": len(columns),
    }
    diagnostics.to_csv(out / "ridge_diagnostics.csv", index=False)
    pd.DataFrame({"characteristic": columns, "weight": weights}).to_csv(out / "weights.csv", index=False)
    pd.DataFrame({"date": test["date"], "payoff_date": test["payoff_date"], "portfolio_return": test_return}).to_parquet(out / "test_returns.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
