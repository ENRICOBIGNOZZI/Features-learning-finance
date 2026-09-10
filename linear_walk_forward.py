from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from linear_managed_baseline import factor_panel, portfolio, ridge_weights
from neural_factors.data import JKPUSData
from neural_factors.train import SplitConfig, annualized_sharpe


def main():
    parser = argparse.ArgumentParser(description="Expanding-window linear managed-portfolio benchmark")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-dir", default="results/causal_v2_linear_managed")
    parser.add_argument("--output-dir", default="results/causal_v2_linear_walkforward")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    split = SplitConfig()
    scaler = data.fit_state_scaler(split.train_start, split.train_end)
    frame = factor_panel(data, "1963-01-31", "2024-11-30", scaler)
    cols = ["intercept", *data.characteristics]
    ridge = float(json.load(open(Path(args.baseline_dir) / "summary.json"))["selected_ridge"])
    rows, folds = [], []
    for year in range(2015, 2025):
        train_end = pd.Timestamp(year=year - 1, month=11, day=30)
        test_start = pd.Timestamp(year=year - 1, month=12, day=31)
        test_end = pd.Timestamp(year=year, month=11, day=30)
        train = frame.loc[frame["date"] <= train_end]
        test = frame.loc[frame["date"].between(test_start, test_end)]
        weights = ridge_weights(train, cols, ridge)
        values = portfolio(test, cols, weights)
        folds.append({"payoff_year": year, "months": len(values), "sharpe": annualized_sharpe(values)})
        for (_, row), value in zip(test.iterrows(), values):
            rows.append({"formation_date": row["date"], "payoff_date": row["payoff_date"], "portfolio_return": float(value), "fold": year})
    returns = pd.DataFrame(rows)
    summary = {
        "selected_ridge_pre2015": ridge,
        "months": len(returns),
        "annualized_sharpe": annualized_sharpe(returns["portfolio_return"].to_numpy()),
    }
    returns.to_parquet(out / "walk_forward_returns.parquet", index=False)
    pd.DataFrame(folds).to_csv(out / "walk_forward_folds.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    print(pd.DataFrame(folds).to_string(index=False))


if __name__ == "__main__":
    main()
