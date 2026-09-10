from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from .data import JKPUSData
from .interpret import load_model
from .train import annualized_sharpe


def collect_weights(
    result_dir: str | Path,
    data: JKPUSData,
    start: str,
    end: str,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, scaler, _ = load_model(result_dir, data, device=device)
    rows = []
    monthly = []
    with torch.no_grad():
        for panel in data.materialize(start, end, scaler):
            x = torch.from_numpy(panel.x).to(device=device, dtype=torch.float32)
            r = torch.from_numpy(panel.r).to(device=device, dtype=torch.float32)
            state = torch.from_numpy(panel.state).to(device=device, dtype=torch.float32)
            out = model(x, state, r)
            weight = out.stock_weights.detach().cpu().numpy()
            monthly.append({
                "date": panel.date,
                "payoff_date": panel.date + pd.offsets.MonthEnd(1),
                "raw_return": float(out.portfolio_return.cpu()),
                "missing_return_share": float(1.0 - np.mean(panel.return_observed)),
            })
            rows.append(pd.DataFrame({
                "date": panel.date,
                "id": panel.ids,
                "weight": weight,
                "ret_fwd": panel.r,
                "return_observed": panel.return_observed,
            }))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(monthly).sort_values("date")


def lagged_vol_scale(
    monthly: pd.DataFrame,
    target_annual_vol: float = 0.10,
    lookback: int = 36,
    min_periods: int = 12,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    out = monthly.sort_values("date").copy()
    lagged = out["raw_return"].shift(1)
    ann_vol = lagged.rolling(lookback, min_periods=min_periods).std(ddof=1) * np.sqrt(12.0)
    expanding = lagged.expanding(min_periods=2).std(ddof=1) * np.sqrt(12.0)
    ann_vol = ann_vol.fillna(expanding)
    scale = target_annual_vol / ann_vol.replace(0.0, np.nan)
    out["scale"] = scale.clip(lower=0.0, upper=max_leverage).fillna(0.0)
    out["vol_target_return"] = out["scale"] * out["raw_return"]
    return out


def apply_scale_to_weights(weights: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    scale = monthly[["date", "scale"]]
    out = weights.merge(scale, on="date", how="left")
    out["scaled_weight"] = out["weight"] * out["scale"]
    return out


def realized_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    weights = weights.sort_values(["date", "id"]).copy()
    dates = list(pd.Index(weights["date"].unique()).sort_values())
    rows = []
    prev_post = {}
    for date in dates:
        block = weights.loc[weights["date"] == date]
        current = dict(zip(block["id"], block["scaled_weight"]))
        ids = set(current) | set(prev_post)
        trade_notional = sum(abs(current.get(i, 0.0) - prev_post.get(i, 0.0)) for i in ids)
        rows.append({
            "date": pd.Timestamp(date),
            "trade_notional": float(trade_notional),
            "turnover_half_l1": float(0.5 * trade_notional),
            "gross_exposure": float(block["scaled_weight"].abs().sum()),
            "long_exposure": float(block["scaled_weight"].clip(lower=0.0).sum()),
            "short_exposure": float(-block["scaled_weight"].clip(upper=0.0).sum()),
            "net_exposure": float(block["scaled_weight"].sum()),
        })
        portfolio_return = float(
            np.sum(block["scaled_weight"].to_numpy(dtype=float) * block["ret_fwd"].to_numpy(dtype=float))
        )
        nav_growth = 1.0 + portfolio_return
        if nav_growth <= 1e-8:
            nav_growth = 1e-8
        prev_post = {
            i: float(w * (1.0 + r) / nav_growth)
            for i, w, r in zip(block["id"], block["scaled_weight"], block["ret_fwd"])
        }
    return pd.DataFrame(rows)


def add_transaction_costs(
    monthly: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    out = monthly.merge(turnover, on="date", how="left")
    out["trading_cost"] = out["trade_notional"].fillna(0.0) * (cost_bps / 1e4)
    out[f"net_return_{cost_bps:g}bps"] = out["vol_target_return"] - out["trading_cost"]
    return out


def performance_summary(frame: pd.DataFrame, return_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in return_columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue
        rows.append({
            "series": column,
            "months": len(values),
            "annualized_sharpe": annualized_sharpe(values),
            "annualized_mean": float(12.0 * values.mean()),
            "annualized_vol": float(np.sqrt(12.0) * values.std(ddof=1)),
            "worst_month": float(values.min()),
        })
    return pd.DataFrame(rows)


def analyze_investability(
    result_dir: str | Path,
    data_dir: str | Path,
    start: str,
    end: str,
    cost_bps: tuple[float, ...] = (5.0, 10.0, 25.0),
    target_annual_vol: float = 0.10,
    vol_lookback: int = 36,
    borrow_bps_annual: tuple[float, ...] = (100.0, 300.0),
) -> dict[str, pd.DataFrame]:
    result_dir = Path(result_dir)
    data = JKPUSData(data_dir)
    start_ts = pd.Timestamp(start)
    # Build risk estimates and carried positions from pre-test history so the
    # first reported month is not treated as a fresh portfolio launch.
    history_start = (start_ts - pd.offsets.MonthEnd(vol_lookback + 1)).strftime("%Y-%m-%d")
    weights, monthly = collect_weights(result_dir, data, history_start, end)
    monthly = lagged_vol_scale(
        monthly, target_annual_vol=target_annual_vol, lookback=vol_lookback
    )
    weights = apply_scale_to_weights(weights, monthly)
    turnover = realized_turnover(weights)
    missing_stress = (
        weights.loc[~weights["return_observed"].astype(bool)]
        .assign(missing_stress=lambda x: 0.30 * x["scaled_weight"].abs())
        .groupby("date", as_index=False)["missing_stress"].sum()
    )
    analyzed = monthly.merge(turnover, on="date", how="left")
    analyzed = analyzed.merge(missing_stress, on="date", how="left")
    analyzed["missing_stress"] = analyzed["missing_stress"].fillna(0.0)
    analyzed["return_missing30pct_adverse"] = (
        analyzed["vol_target_return"] - analyzed["missing_stress"]
    )
    for bps in cost_bps:
        trading_cost = analyzed["trade_notional"].fillna(0.0) * (bps / 1e4)
        analyzed[f"net_return_{bps:g}bps"] = analyzed["vol_target_return"] - trading_cost
    for borrow in borrow_bps_annual:
        borrow_cost = analyzed["short_exposure"].fillna(0.0) * (borrow / 1e4) / 12.0
        analyzed[f"net_return_10bps_borrow{borrow:g}bps"] = (
            analyzed["vol_target_return"]
            - analyzed["trade_notional"].fillna(0.0) * (10.0 / 1e4)
            - borrow_cost
        )
    report_mask = analyzed["date"] >= start_ts
    analyzed_report = analyzed.loc[report_mask].reset_index(drop=True)
    weights_report = weights.loc[weights["date"] >= start_ts].reset_index(drop=True)
    columns = ["raw_return", "vol_target_return", "return_missing30pct_adverse"]
    columns += [f"net_return_{bps:g}bps" for bps in cost_bps]
    columns += [f"net_return_10bps_borrow{borrow:g}bps" for borrow in borrow_bps_annual]
    return {
        "monthly": analyzed_report,
        "weights": weights_report,
        "summary": performance_summary(analyzed_report, columns),
    }
