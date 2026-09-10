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
    # These are formation dates. ret_exc_lead1m is realized one month later,
    # so adjacent payoff samples require a one-month formation-date offset.
    # Payoffs: train through Dec-2004, validation Jan-2005..Dec-2014,
    # test Jan-2015..Dec-2024.
    train_start: str = "1963-01-31"
    train_end: str = "2004-11-30"
    val_start: str = "2004-12-31"
    val_end: str = "2014-11-30"
    test_start: str = "2014-12-31"
    test_end: str = "2024-11-30"


@dataclass
class TrainConfig:
    hidden: int = 64
    factors: int = 16
    context_heads: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    batch_months: int = 12
    warmup_epochs: int = 3
    objective_mode: str = "global"
    train_stocks_per_month: int = 0
    grad_clip: float = 2.0
    min_lr: float = 2e-5
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


def tangency_score(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-free objective monotone in the positive Sharpe ratio."""
    mean = returns.mean()
    second_moment = returns.square().mean()
    return mean / torch.sqrt(second_moment + eps)


def _global_objective_step(
    model: NeuralFactorModel,
    panels: list[MonthPanel],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    *,
    response_one: bool = False,
    batch_months: int = 12,
) -> float:
    """One exact full-sample gradient step using vectorized temporal blocks.

    The first pass estimates current global return moments. The second pass
    applies the analytic derivative of the global tangency criterion while
    freeing each padded block immediately.
    """
    model.eval()
    realized: list[float] = []
    with torch.no_grad():
        for start in range(0, len(panels), batch_months):
            chunk = panels[start:start + batch_months]
            x, state, r, offsets = _pack_panels(chunk, device)
            out = model.forward_ragged(x, state, r, offsets)
            realized.extend(out.portfolio_return.detach().cpu().numpy().tolist())
    values = np.asarray(realized, dtype=np.float64)
    n = max(len(values), 1)
    mu = float(values.mean())
    m2 = float(np.mean(values * values))
    denom = max(m2 + 1e-8, 1e-8)
    score = mu / np.sqrt(denom)
    if response_one:
        coefficients = 2.0 * (values - 1.0) / n
    else:
        coefficients = -(1.0 / (n * np.sqrt(denom)) - mu * values / (n * denom ** 1.5))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(panels), batch_months):
        chunk = panels[start:start + batch_months]
        x, state, r, offsets = _pack_panels(chunk, device)
        out = model.forward_ragged(x, state, r, offsets)
        coeff = torch.as_tensor(
            coefficients[start:start + len(chunk)], device=device, dtype=out.portfolio_return.dtype
        )
        (out.portfolio_return * coeff).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(score)

def _to_device(panel: MonthPanel, device: torch.device):
    x = torch.from_numpy(panel.x).to(device=device, dtype=torch.float32)
    r = torch.from_numpy(panel.r).to(device=device, dtype=torch.float32)
    state = torch.from_numpy(panel.state).to(device=device, dtype=torch.float32)
    return x, state, r


def _subsample_panels(
    panels: list[MonthPanel],
    max_stocks: int,
    rng: np.random.Generator,
) -> list[MonthPanel]:
    if max_stocks <= 0:
        return panels
    sampled = []
    for panel in panels:
        n = len(panel.r)
        if n <= max_stocks:
            sampled.append(panel)
            continue
        idx = np.sort(rng.choice(n, size=max_stocks, replace=False))
        sampled.append(MonthPanel(
            date=panel.date, x=panel.x[idx], r=panel.r[idx], ids=panel.ids[idx],
            me=panel.me[idx], state=panel.state, return_observed=panel.return_observed[idx],
        ))
    return sampled


def _pack_panels(panels: list[MonthPanel], device: torch.device):
    """Concatenate a temporal block without padding its monthly cross-sections."""
    if not panels:
        raise ValueError("cannot pack an empty panel list")
    lengths = [len(panel.r) for panel in panels]
    offsets = np.r_[0, np.cumsum(lengths)].astype(int).tolist()
    x = np.concatenate([panel.x for panel in panels], axis=0)
    r = np.concatenate([panel.r for panel in panels], axis=0)
    state = np.stack([panel.state for panel in panels], axis=0)
    return (
        torch.from_numpy(x).to(device=device, dtype=torch.float32),
        torch.from_numpy(state).to(device=device, dtype=torch.float32),
        torch.from_numpy(r).to(device=device, dtype=torch.float32),
        offsets,
    )


def _pad_panels(panels: list[MonthPanel], device: torch.device):
    """Pad a short temporal block so all stock-level NN work is vectorized."""
    if not panels:
        raise ValueError("cannot pad an empty panel list")
    batch = len(panels)
    max_n = max(len(panel.r) for panel in panels)
    d = panels[0].x.shape[1]
    state_dim = len(panels[0].state)
    x = np.zeros((batch, max_n, d), dtype=np.float16)
    r = np.zeros((batch, max_n), dtype=np.float32)
    mask = np.zeros((batch, max_n), dtype=bool)
    state = np.zeros((batch, state_dim), dtype=np.float32)
    for row, panel in enumerate(panels):
        n = len(panel.r)
        x[row, :n] = panel.x
        r[row, :n] = panel.r
        mask[row, :n] = True
        state[row] = panel.state
    return (
        torch.from_numpy(x).to(device=device, dtype=torch.float32),
        torch.from_numpy(state).to(device=device, dtype=torch.float32),
        torch.from_numpy(r).to(device=device, dtype=torch.float32),
        torch.from_numpy(mask).to(device=device),
    )


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
    batch_months: int = 12,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    panels = data.materialize(start, end, scaler)
    with torch.no_grad():
        for offset in range(0, len(panels), batch_months):
            chunk = panels[offset:offset + batch_months]
            x, state, r, offsets = _pack_panels(chunk, device)
            out = model.forward_ragged(x, state, r, offsets)
            portfolio = out.portfolio_return.detach().cpu().numpy()
            factor_returns = out.factor_returns.detach().cpu().numpy()
            allocators = out.allocator.detach().cpu().numpy()
            for row_index, panel in enumerate(chunk):
                value = float(portfolio[row_index])
                row = {
                    "date": panel.date,
                    "payoff_date": panel.date + pd.offsets.MonthEnd(1),
                    "portfolio_return": value,
                    "response_one_loss": float((1.0 - value) ** 2),
                    "missing_return_share": float(1.0 - np.mean(panel.return_observed)),
                }
                for j, factor_value in enumerate(factor_returns[row_index]):
                    row[f"factor_{j:02d}"] = float(factor_value)
                for j, allocation in enumerate(allocators[row_index]):
                    row[f"allocator_{j:02d}"] = float(allocation)
                rows.append(row)
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = {
        "loss": float(frame["response_one_loss"].mean()),
        "sharpe": annualized_sharpe(frame["portfolio_return"].to_numpy()),
        "mean_monthly": float(frame["portfolio_return"].mean()),
        "vol_monthly": float(frame["portfolio_return"].std(ddof=1)),
        "months": int(len(frame)),
        "mean_missing_return_share": float(frame["missing_return_share"].mean()),
        "max_missing_return_share": float(frame["missing_return_share"].max()),
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
    linear_characteristics: bool = False,
    conditional_scores: bool = False,
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
        linear_characteristics=linear_characteristics,
        conditional_scores=conditional_scores,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=config.min_lr
    )
    train_panels = data.materialize(split.train_start, split.train_end, scaler)
    history = []
    best_sharpe = -float("inf")
    best_state = None
    stale = 0

    for epoch in range(config.epochs):
        sample_rng = np.random.default_rng(config.seed * 100_003 + epoch)
        epoch_panels = _subsample_panels(
            train_panels, config.train_stocks_per_month, sample_rng
        )
        if config.objective_mode == "global":
            train_score = _global_objective_step(
                model, epoch_panels, optimizer, device, config.grad_clip,
                response_one=epoch < config.warmup_epochs,
                batch_months=config.batch_months,
            )
            train_scores = [train_score]
        else:
            model.train()
            train_scores = []
            for offset in range(0, len(epoch_panels), config.batch_months):
                chunk = epoch_panels[offset:offset + config.batch_months]
                x, state, r, offsets = _pack_panels(chunk, device)
                optimizer.zero_grad(set_to_none=True)
                out = model.forward_ragged(x, state, r, offsets)
                returns = out.portfolio_return
                score = tangency_score(returns)
                objective = (1.0 - returns).square().mean() if epoch < config.warmup_epochs else -score
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_scores.append(float(score.detach().cpu()))

        val_metrics, _ = evaluate(
            model, data, split.val_start, split.val_end, scaler, device
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1,
            "train_tangency_score": float(np.mean(train_scores)),
            "val_loss": val_metrics["loss"],
            "val_sharpe": val_metrics["sharpe"],
            "lr": current_lr,
        }
        history.append(row)
        print(row, flush=True)
        scheduler.step(val_metrics["sharpe"])

        if val_metrics["sharpe"] > best_sharpe + 1e-4:
            best_sharpe = val_metrics["sharpe"]
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
                "linear_characteristics": linear_characteristics,
                "conditional_scores": conditional_scores,
                "characteristics": data.characteristics,
                "state_mean": scaler[0].tolist(),
                "state_std": scaler[1].tolist(),
                "device": str(device),
            },
            stream,
            indent=2,
        )
    return model, scaler, device
