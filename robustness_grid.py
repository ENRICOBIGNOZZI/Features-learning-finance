from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_factors.analysis import allocation_dimension, fit_canonical_basis
from neural_factors.data import JKPUSData
from neural_factors.train import SplitConfig, TrainConfig, evaluate, fit_model


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Kmax x seed robustness battery")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/causal_k_grid")
    parser.add_argument("--preset", choices=["pilot", "paper"], default="paper")
    parser.add_argument("--k-values", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--context-heads", type=int, default=2)
    parser.add_argument("--batch-months", type=int, default=12)
    parser.add_argument("--train-stocks-per-month", type=int, default=1024)
    parser.add_argument("--objective-mode", choices=["block", "global"], default="block")
    return parser.parse_args()


def split_for(preset: str) -> SplitConfig:
    if preset == "paper":
        return SplitConfig()
    return SplitConfig(
        train_start="2000-01-31", train_end="2014-11-30",
        val_start="2014-12-31", val_end="2019-11-30",
        test_start="2019-12-31", test_end="2024-11-30",
    )


def completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("kmax")
    rows = []
    for k, block in grouped:
        rows.append({
            "kmax": int(k), "runs": len(block),
            "test_sharpe_mean": block["test_sharpe"].mean(),
            "test_sharpe_median": block["test_sharpe"].median(),
            "test_sharpe_std": block["test_sharpe"].std(ddof=1),
            "test_sharpe_p10": block["test_sharpe"].quantile(0.10),
            "test_sharpe_p90": block["test_sharpe"].quantile(0.90),
            "val_sharpe_mean": block["val_sharpe"].mean(),
            "allocation_rank_mean": block["allocation_rank"].mean(),
            "first_direction_share_mean": block["first_direction_share"].mean(),
        })
    return pd.DataFrame(rows).sort_values("kmax")


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "robustness_grid.csv"
    rows = completed_rows(result_path)
    completed = {(int(row["kmax"]), int(row["seed"])) for row in rows}
    data = JKPUSData(args.data_dir)
    split = split_for(args.preset)
    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)

    for k in k_values:
        for seed in seeds:
            if (k, seed) in completed:
                continue
            run_dir = root / f"k{k:02d}_seed{seed:02d}"
            config = TrainConfig(
                hidden=args.hidden, factors=k, context_heads=args.context_heads,
                lr=1e-3, weight_decay=5e-4, epochs=args.epochs, patience=5,
                batch_months=args.batch_months, warmup_epochs=2,
                objective_mode=args.objective_mode,
                train_stocks_per_month=args.train_stocks_per_month, seed=seed,
            )
            model, scaler, device = fit_model(
                data, split, config, run_dir, device_name=args.device,
                use_context=True, use_state=True,
            )
            train_metrics, train_frame = evaluate(
                model, data, split.train_start, split.train_end, scaler, device
            )
            val_metrics, _ = evaluate(
                model, data, split.val_start, split.val_end, scaler, device
            )
            test_metrics, test_frame = evaluate(
                model, data, split.test_start, split.test_end, scaler, device
            )
            test_frame.to_parquet(run_dir / "monthly_test.parquet", index=False)
            basis = fit_canonical_basis(train_frame)
            dims = allocation_dimension(basis.economic_eigenvalues)
            values = np.maximum(basis.economic_eigenvalues, 0.0)
            shares = values / max(values.sum(), 1e-12)
            row = {
                "kmax": k, "seed": seed,
                "train_sharpe": train_metrics["sharpe"],
                "val_sharpe": val_metrics["sharpe"],
                "test_sharpe": test_metrics["sharpe"],
                "allocation_rank": dims["entropy_rank"],
                "participation_ratio": dims["participation_ratio"],
                "first_direction_share": float(shares[0]),
                "top2_share": float(shares[:2].sum()),
                "test_missing_return_share": test_metrics["mean_missing_return_share"],
            }
            rows.append(row)
            results = pd.DataFrame(rows).sort_values(["kmax", "seed"])
            results.to_csv(result_path, index=False)
            summarize(results).to_csv(root / "robustness_summary.csv", index=False)
            print("ROBUSTNESS", row, flush=True)
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    results = pd.DataFrame(rows).sort_values(["kmax", "seed"])
    summary = summarize(results)
    summary.to_csv(root / "robustness_summary.csv", index=False)
    print("\nRobustness summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
