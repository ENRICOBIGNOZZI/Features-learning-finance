from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from neural_factors.data import JKPUSData
from neural_factors.train import SplitConfig, TrainConfig, evaluate, fit_model


def parse_args():
    parser = argparse.ArgumentParser(description="Neural portfolio architecture ablations")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/ablations_v5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--preset", choices=["pilot", "paper"], default="pilot")
    parser.add_argument("--batch-months", type=int, default=36)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--context-heads", type=int, default=2)
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


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    split = split_for(args.preset)
    variants = {
        "full": (True, True, False, False, False),
        "no_state": (True, False, False, False, False),
        "no_context": (False, True, False, False, False),
        "static": (False, False, True, False, False),
        "linear_characteristics": (True, True, False, True, False),
        "direct_conditional_nn": (True, True, True, False, True),
    }
    completed_path = root / "ablation_summary.csv"
    rows = pd.read_csv(completed_path).to_dict("records") if completed_path.exists() else []
    completed = {(str(row["model"]), int(row["seed"])) for row in rows}

    for name, spec in variants.items():
        use_context, use_state, static_allocator, linear_characteristics, conditional_scores = spec
        for seed in range(args.seeds):
            if (name, seed) in completed:
                continue
            run_dir = root / f"{name}_seed{seed:02d}"
            factor_count = 1 if conditional_scores else args.factors
            config = TrainConfig(
                hidden=args.hidden, factors=factor_count, context_heads=args.context_heads,
                lr=1e-3, weight_decay=5e-4, epochs=args.epochs,
                patience=5, batch_months=args.batch_months,
                objective_mode=args.objective_mode,
                train_stocks_per_month=args.train_stocks_per_month, seed=seed,
            )
            model, scaler, device = fit_model(
                data, split, config, run_dir, device_name=args.device,
                use_context=use_context, use_state=use_state,
                static_allocator=static_allocator,
                linear_characteristics=linear_characteristics,
                conditional_scores=conditional_scores,
            )
            val_metrics, _ = evaluate(model, data, split.val_start, split.val_end, scaler, device)
            test_metrics, test_frame = evaluate(model, data, split.test_start, split.test_end, scaler, device)
            test_frame.to_parquet(run_dir / "monthly_test.parquet", index=False)
            row = {
                "model": name, "seed": seed,
                "val_sharpe": val_metrics["sharpe"],
                "test_sharpe": test_metrics["sharpe"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(completed_path, index=False)
            print(row, flush=True)
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    results = pd.DataFrame(rows)
    summary = results.groupby("model").agg(
        seeds=("seed", "count"),
        val_sharpe_mean=("val_sharpe", "mean"),
        val_sharpe_std=("val_sharpe", "std"),
        test_sharpe_mean=("test_sharpe", "mean"),
        test_sharpe_std=("test_sharpe", "std"),
        test_sharpe_median=("test_sharpe", "median"),
    ).reset_index()
    summary.to_csv(root / "ablation_group_summary.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
