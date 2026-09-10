from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_factors.data import JKPUSData, MonthPanel
from neural_factors.model import NeuralFactorModel
from neural_factors.train import (
    SplitConfig,
    TrainConfig,
    annualized_sharpe,
    choose_device,
    fit_model,
    tangency_score,
    _pack_panels,
    _subsample_panels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Recursive expanding-window OOS test")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/walk_forward_v5")
    parser.add_argument("--initial-result-dir")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--initial-epochs", type=int, default=30)
    parser.add_argument("--pretest-refit-epochs", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--batch-months", type=int, default=36)
    parser.add_argument("--train-stocks-per-month", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_device(panel: MonthPanel, device: torch.device):
    return (
        torch.from_numpy(panel.x).to(device=device, dtype=torch.float32),
        torch.from_numpy(panel.state).to(device=device, dtype=torch.float32),
        torch.from_numpy(panel.r).to(device=device, dtype=torch.float32),
    )


def payoff_date(panel: MonthPanel) -> pd.Timestamp:
    return pd.Timestamp(panel.date) + pd.offsets.MonthEnd(1)


def panel_returns(model, panels, device) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for panel in panels:
            x, state, r = to_device(panel, device)
            values.append(float(model(x, state, r).portfolio_return.cpu()))
    return np.asarray(values, dtype=float)


def update_on_expanding_window(
    model: NeuralFactorModel,
    panels: list[MonthPanel],
    device: torch.device,
    epochs: int,
    batch_months: int,
    lr: float,
    weight_decay: float,
    train_stocks_per_month: int = 0,
    seed: int = 0,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed * 100_003 + epoch)
        epoch_panels = _subsample_panels(panels, train_stocks_per_month, rng)
        model.train()
        scores = []
        for offset in range(0, len(epoch_panels), batch_months):
            chunk = epoch_panels[offset:offset + batch_months]
            x, state, r, offsets = _pack_panels(chunk, device)
            optimizer.zero_grad(set_to_none=True)
            returns = model.forward_ragged(x, state, r, offsets).portfolio_return
            score = tangency_score(returns)
            (-score).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scores.append(float(score.detach().cpu()))
        history.append({
            "epoch": epoch + 1,
            "train_tangency_score": float(np.mean(scores)),
        })
    return history

def load_initial_model(result_dir: Path, data: JKPUSData, device: torch.device):
    config = json.loads((result_dir / "config.json").read_text())
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
    ).to(device)
    state = torch.load(result_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    scaler = (
        np.asarray(config["state_mean"], dtype=np.float32),
        np.asarray(config["state_std"], dtype=np.float32),
    )
    return model, scaler


def fit_initial_model(args, data: JKPUSData, output_dir: Path, device: torch.device):
    split = SplitConfig()
    config = TrainConfig(
        hidden=args.hidden,
        factors=args.factors,
        context_heads=args.context_heads,
        lr=1e-3,
        weight_decay=args.weight_decay,
        epochs=args.initial_epochs,
        patience=8,
        batch_months=args.batch_months,
        objective_mode="block", train_stocks_per_month=args.train_stocks_per_month,
        seed=args.seed,
    )
    model, scaler, _ = fit_model(
        data, split, config, output_dir / "initial_model",
        device_name=str(device), use_context=True, use_state=True,
    )
    return model, scaler


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    device = choose_device(args.device)

    if args.initial_result_dir:
        model, scaler = load_initial_model(Path(args.initial_result_dir), data, device)
    else:
        model, scaler = fit_initial_model(args, data, output_dir, device)

    all_panels = data.materialize(
        "1963-01-01", f"{args.end_year}-11-30", scaler, cache=True
    )
    oos_rows = []
    fold_rows = []
    update_history = []

    # Hyperparameters were selected using the pre-2015 validation period. Once
    # frozen, those observations may be used for estimation before the first
    # true OOS payoff in January 2015.
    pretest_cutoff = pd.Timestamp(year=args.start_year - 1, month=11, day=30)
    pretest_panels = [p for p in all_panels if pd.Timestamp(p.date) <= pretest_cutoff]
    if args.pretest_refit_epochs > 0:
        history = update_on_expanding_window(
            model, pretest_panels, device, args.pretest_refit_epochs,
            args.batch_months, args.lr, args.weight_decay,
            args.train_stocks_per_month, args.seed + 10_000,
        )
        for row in history:
            update_history.append({"test_year": args.start_year, "phase": "pretest_refit", **row})

    for year in range(args.start_year, args.end_year + 1):
        if year > args.start_year:
            available_cutoff = pd.Timestamp(year=year - 1, month=11, day=30)
            expanding = [p for p in all_panels if p.date <= available_cutoff]
            history = update_on_expanding_window(
                model, expanding, device, args.update_epochs,
                args.batch_months, args.lr, args.weight_decay,
                args.train_stocks_per_month, args.seed + year,
            )
            for row in history:
                update_history.append({"test_year": year, "phase": "annual_update", **row})

        formation_start = pd.Timestamp(year=year - 1, month=12, day=31)
        formation_end = pd.Timestamp(year=year, month=11, day=30)
        test_panels = [
            p for p in all_panels
            if formation_start <= pd.Timestamp(p.date) <= formation_end
        ]
        values = panel_returns(model, test_panels, device)
        fold_sr = annualized_sharpe(values)
        fold_rows.append({
            "payoff_year": year,
            "formation_start": formation_start,
            "formation_end": formation_end,
            "months": len(values),
            "sharpe": float(fold_sr),
        })
        for panel, value in zip(test_panels, values):
            oos_rows.append({
                "formation_date": pd.Timestamp(panel.date),
                "payoff_date": payoff_date(panel),
                "portfolio_return": float(value),
                "fold": year,
                "missing_return_share": float(1.0 - np.mean(panel.return_observed)),
            })
        print(f"payoff year {year}: SR={fold_sr:.3f}", flush=True)

    oos = pd.DataFrame(oos_rows).sort_values("payoff_date").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    summary = {
        "payoff_start": str(oos["payoff_date"].min().date()),
        "payoff_end": str(oos["payoff_date"].max().date()),
        "months": int(len(oos)),
        "annualized_sharpe": annualized_sharpe(oos["portfolio_return"].to_numpy()),
        "annualized_mean": float(12.0 * oos["portfolio_return"].mean()),
        "annualized_vol": float(np.sqrt(12.0) * oos["portfolio_return"].std(ddof=1)),
        "mean_missing_return_share": float(oos["missing_return_share"].mean()),
    }
    oos.to_parquet(output_dir / "walk_forward_returns.parquet", index=False)
    folds.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    pd.DataFrame(update_history).to_csv(
        output_dir / "walk_forward_update_history.csv", index=False
    )
    (output_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWalk-forward summary", summary)


if __name__ == "__main__":
    main()
