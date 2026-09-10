from __future__ import annotations

import argparse
from pathlib import Path
import json

from neural_factors.analysis import save_canonical_analysis, economic_dimension
from neural_factors.data import JKPUSData
from neural_factors.policy_spectrum import save_policy_spectrum
from neural_factors.train import SplitConfig, TrainConfig, fit_model, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end neural factor discovery on JKP USA")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--preset", choices=["pilot", "paper"], default="pilot")
    parser.add_argument("--variant", choices=["full", "no_state", "no_context", "static", "linear", "direct"], default="full")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-months", type=int, default=12)
    parser.add_argument("--objective-mode", choices=["block", "global"], default="block")
    parser.add_argument("--train-stocks-per-month", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_for(preset: str) -> SplitConfig:
    if preset == "pilot":
        return SplitConfig(
            train_start="2000-01-31", train_end="2014-11-30",
            val_start="2014-12-31", val_end="2019-11-30",
            test_start="2019-12-31", test_end="2024-11-30",
        )
    return SplitConfig()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    data = JKPUSData(args.data_dir)
    split = split_for(args.preset)
    conditional_scores = args.variant == "direct"
    factor_count = 1 if conditional_scores else args.factors
    config = TrainConfig(
        hidden=args.hidden,
        factors=factor_count,
        context_heads=args.context_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_months=args.batch_months,
        objective_mode=args.objective_mode,
        train_stocks_per_month=args.train_stocks_per_month,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
    )
    use_context = args.variant != "no_context"
    use_state = args.variant != "no_state"
    static_allocator = args.variant == "static"
    linear_characteristics = args.variant == "linear"
    if conditional_scores:
        static_allocator = True
    model, scaler, device = fit_model(
        data=data,
        split=split,
        config=config,
        output_dir=output_dir,
        device_name=args.device,
        use_context=use_context,
        use_state=use_state,
        static_allocator=static_allocator,
        linear_characteristics=linear_characteristics,
        conditional_scores=conditional_scores,
    )

    frames = {}
    metrics = {}
    ranges = {
        "train": (split.train_start, split.train_end),
        "val": (split.val_start, split.val_end),
        "test": (split.test_start, split.test_end),
    }
    for name, (start, end) in ranges.items():
        split_metrics, frame = evaluate(model, data, start, end, scaler, device)
        metrics[name] = split_metrics
        frames[name] = frame
        frame.to_parquet(output_dir / f"monthly_{name}.parquet", index=False)

    basis = save_canonical_analysis(frames["train"], frames, output_dir / "factors")
    metrics["economic_dimension"] = economic_dimension(basis.economic_eigenvalues)
    metrics["policy_dimension"] = save_policy_spectrum(
        model, data, split.train_start, split.train_end, scaler, device,
        output_dir / "policy", batch_months=args.batch_months,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print("\nFinal metrics")
    for name in ("train", "val", "test"):
        print(name, metrics[name])
    print("economic dimension", metrics["economic_dimension"])


if __name__ == "__main__":
    main()
