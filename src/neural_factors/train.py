from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch

from .data import JKPUSData, MonthPanel
from .model import NeuralFactorModel


@dataclass
class SplitConfig:
    train_start: str = "1963-01-31"
    train_end: str = "2004-12-31"
    val_start: str = "2005-01-31"
    val_end: str = "2014-12-31"
    test_start: str = "2015-01-31"
    test_end: str = "2024-12-31"


@dataclass
class TrainConfig:
    hidden: int = 64
    factors: int = 16
    context_heads: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 6
    grad_accum_months: int = 8
    grad_clip: float = 5.0
    seed: int = 42


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def annualized_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(12.0) * returns.mean() / returns.std(ddof=1))


def _to_device(panel: MonthPanel, device: torch.device):
    x = torch.from_numpy(panel.x).to(device)
    r = torch.from_numpy(panel.r).to(device)
    state = torch.from_numpy(panel.state).to(device)
    return x, state, r


def _iter_panels(
    data: JKPUSData,
    years: list[int],
    start: str,
    end: str,
    scaler,
    rng: random.Random | None = None,
):
    year_order = list(years)
    if rng is not None:
        rng.shuffle(year_order)
    for year in year_order:
        panels = list(data.iter_year(year, start, end, scaler))
        if rng is not None:
            rng.shuffle(panels)
        yield from panels


def evaluate(
    model: NeuralFactorModel,
    data: JKPUSData,
    start: str,
    end: str,
    scaler,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    years = data.years(start, end)
    with torch.no_grad():
        for panel in _iter_panels(data, years, start, end, scaler):
            x, state, r = _to_device(panel, device)
            out = model(x, state, r)
            row = {
                "date": panel.date,
                "portfolio_return": float(out.portfolio_return.cpu()),
                "response_one_loss": float((1.0 - out.portfolio_return).square().cpu()),
            }
            for j, value in enumerate(out.factor_returns.cpu().numpy()):
                row[f"factor_{j:02d}"] = float(value)
            for j, value in enumerate(out.allocator.cpu().numpy()):
                row[f"allocator_{j:02d}"] = float(value)
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = {
        "loss": float(frame["response_one_loss"].mean()),
        "sharpe": annualized_sharpe(frame["portfolio_return"].to_numpy()),
        "mean_monthly": float(frame["portfolio_return"].mean()),
        "vol_monthly": float(frame["portfolio_return"].std(ddof=1)),
        "months": int(len(frame)),
    }
    return metrics, frame


def fit_model(
    data: JKPUSData,
    split: SplitConfig,
    config: TrainConfig,
    output_dir: str | Path,
    device_name: str = "auto",
    use_context: bool = True,
    use_state: bool = True,
    static_allocator: bool = False,
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = choose_device(device_name)
    scaler = data.fit_state_scaler(split.train_start, split.train_end)

    model = NeuralFactorModel(
        input_dim=data.input_dim,
        state_dim=len(data.state_columns),
        hidden=config.hidden,
        factors=config.factors,
        context_heads=config.context_heads,
        use_context=use_context,
        use_state=use_state,
        static_allocator=static_allocator,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_years = data.years(split.train_start, split.train_end)
    rng = random.Random(config.seed)
    history = []
    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        pending = 0
        panels = _iter_panels(
            data, train_years, split.train_start, split.train_end, scaler, rng
        )
        for panel in panels:
            x, state, r = _to_device(panel, device)
            out = model(x, state, r)
            loss = (1.0 - out.portfolio_return).square()
            (loss / config.grad_accum_months).backward()
            losses.append(float(loss.detach().cpu()))
            pending += 1
            if pending == config.grad_accum_months:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_metrics, _ = evaluate(
            model, data, split.val_start, split.val_end, scaler, device
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_metrics["loss"],
            "val_sharpe": val_metrics["sharpe"],
        }
        history.append(row)
        print(row, flush=True)

        if val_metrics["loss"] < best_loss - 1e-6:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "split": asdict(split),
                "train": asdict(config),
                "use_context": use_context,
                "use_state": use_state,
                "static_allocator": static_allocator,
                "characteristics": data.characteristics,
                "state_mean": scaler[0].tolist(),
                "state_std": scaler[1].tolist(),
                "device": str(device),
            },
            stream,
            indent=2,
        )
    return model, scaler, device
