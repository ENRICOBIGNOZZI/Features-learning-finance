from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from neural_factors.benchmarks import (
    benchmark_model_columns,
    default_benchmark_panel,
    download_stambaugh_yuan,
    nw_spanning,
    oos_hedged_factor,
    oos_marginal_sharpe,
    payoff_month_from_formation,
)
from neural_factors.train import annualized_sharpe


def parse_args():
    parser = argparse.ArgumentParser(description="Spanning and marginal-Sharpe analysis")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--cache-dir", default="data/benchmarks")
    return parser.parse_args()


def load_canonical(result_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(result_dir / "factors" / f"canonical_{split}.parquet")
    out = frame[["date", "canonical_factor_00"]].rename(
        columns={"canonical_factor_00": "learned_factor"}
    ).copy()
    # JKP's ret_exc_lead1m attached to formation month t is the payoff in t+1.
    # Benchmark factors are dated by the month in which their return is realized.
    out["formation_date"] = out["date"]
    out["date"] = payoff_month_from_formation(out["formation_date"])
    return out


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = result_dir / "spanning"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_canonical(result_dir, "train")
    val = load_canonical(result_dir, "val")
    test = load_canonical(result_dir, "test")
    estimation_factor = pd.concat([train, val], ignore_index=True)["learned_factor"]
    estimation_ann_vol = float(np.sqrt(12.0) * estimation_factor.std(ddof=1))
    factor_scale = 0.10 / max(estimation_ann_vol, 1e-12)
    for frame in (train, val, test):
        frame["learned_factor"] *= factor_scale
    all_dates = pd.concat([train[["date"]], val[["date"]], test[["date"]]])
    start, end = all_dates["date"].min(), all_dates["date"].max()
    benchmarks = default_benchmark_panel(
        str(start.date()), str(end.date()), cache_dir=args.cache_dir
    )

    estimation = pd.concat([train, val], ignore_index=True).merge(
        benchmarks, on="date", how="inner"
    )
    test_panel = test.merge(benchmarks, on="date", how="inner")

    spanning_rows = []
    marginal_rows = []
    hedged_rows = []
    for name, columns in benchmark_model_columns().items():
        available = [column for column in columns if column in test_panel.columns]
        common = test_panel[["learned_factor", *available]].dropna()
        if len(common) < max(24, len(available) + 8):
            continue
        row = {"model": name, **nw_spanning(test_panel, "learned_factor", available)}
        spanning_rows.append(row)
        marginal = oos_marginal_sharpe(
            estimation, test_panel, "learned_factor", available
        )
        marginal_rows.append({"model": name, **marginal})
        hedged = oos_hedged_factor(
            estimation, test_panel, "learned_factor", available
        )
        hedged_rows.append({
            "model": name,
            "hedged_sharpe": annualized_sharpe(hedged["hedged_factor"].to_numpy()),
            "raw_sharpe": annualized_sharpe(hedged["raw_factor"].to_numpy()),
            "months": len(hedged),
        })

    # The original Stambaugh--Yuan file ends in 2016. It provides only
    # 24 months inside the 2015--2024 evaluation period, so we do not mix its
    # historical overlap with the main OOS spanning table. Full-period JKP
    # mispricing factors are used instead.

    spanning = pd.DataFrame(spanning_rows)
    marginal = pd.DataFrame(marginal_rows)
    hedged = pd.DataFrame(hedged_rows)
    spanning.to_csv(output_dir / "spanning_alpha.csv", index=False)
    marginal.to_csv(output_dir / "marginal_sharpe.csv", index=False)
    hedged.to_csv(output_dir / "hedged_factor_sharpe.csv", index=False)
    benchmarks.to_parquet(output_dir / "benchmark_panel.parquet", index=False)

    print("\nSpanning alphas")
    print(spanning.to_string(index=False))
    print("\nOut-of-sample marginal Sharpe")
    print(marginal.to_string(index=False))
    print("\nHedged learned-factor Sharpe")
    print(hedged.to_string(index=False))


if __name__ == "__main__":
    main()
