from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neural_factors.inference import sharpe_confidence_interval
from neural_factors.train import annualized_sharpe


def parse_args():
    parser = argparse.ArgumentParser(description="Equal-seed portfolio ensembles by K")
    parser.add_argument("--grid-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.grid_dir)
    grid = pd.read_csv(root / "robustness_grid.csv")
    rows = []
    ensemble_frames = []
    for k in sorted(grid["kmax"].unique()):
        seed_frames = []
        for seed in sorted(grid.loc[grid["kmax"] == k, "seed"].astype(int)):
            path = root / f"k{int(k):02d}_seed{seed:02d}" / "monthly_test.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)[["payoff_date", "portfolio_return"]].copy()
            frame = frame.rename(columns={"portfolio_return": f"seed_{seed:02d}"})
            seed_frames.append(frame)
        if not seed_frames:
            continue
        merged = seed_frames[0]
        for frame in seed_frames[1:]:
            merged = merged.merge(frame, on="payoff_date", how="inner")
        seed_columns = [c for c in merged if c.startswith("seed_")]
        merged["ensemble_return"] = merged[seed_columns].mean(axis=1)
        ci = sharpe_confidence_interval(
            merged["ensemble_return"].to_numpy(),
            block_length=12, replications=args.bootstrap_reps, seed=1000 + int(k),
        )
        rows.append({
            "kmax": int(k), "seeds": len(seed_columns),
            "ensemble_sharpe": annualized_sharpe(merged["ensemble_return"].to_numpy()),
            "sharpe_ci_low": ci["ci_low"], "sharpe_ci_high": ci["ci_high"],
        })
        merged.insert(0, "kmax", int(k))
        ensemble_frames.append(merged[["kmax", "payoff_date", "ensemble_return"]])
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "seed_ensemble_summary.csv", index=False)
    if ensemble_frames:
        pd.concat(ensemble_frames, ignore_index=True).to_parquet(
            root / "seed_ensemble_returns.parquet", index=False
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
