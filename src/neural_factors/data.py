from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

META_COLUMNS = {
    "id", "eom", "excntry", "gvkey", "permno", "size_grp", "me",
    "ret_exc_lead1m",
}


@dataclass
class MonthPanel:
    date: pd.Timestamp
    x: np.ndarray
    r: np.ndarray
    ids: np.ndarray
    me: np.ndarray
    state: np.ndarray
    return_observed: np.ndarray


class JKPUSData:
    """Streaming loader for the cleaned JKP USA stock-level panel."""

    def __init__(self, data_dir: str | Path, add_log_me: bool = True):
        self.data_dir = Path(data_dir)
        self.add_log_me = add_log_me
        self.files = self._find_files()
        self.characteristics = self._characteristics()
        self.monthly_state = self._build_monthly_state()
        self._panel_cache: dict[tuple[str, str], list[MonthPanel]] = {}

    def _find_files(self) -> dict[int, Path]:
        files = {}
        for path in sorted(self.data_dir.glob("JKP_USA_clean_*.parquet")):
            try:
                year = int(path.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            files[year] = path
        if not files:
            raise FileNotFoundError(
                f"No JKP_USA_clean_YYYY.parquet files in {self.data_dir.resolve()}"
            )
        return files

    def _characteristics(self) -> list[str]:
        manifest = self.data_dir / "cleaning_manifest.json"
        if manifest.exists():
            with manifest.open(encoding="utf-8") as stream:
                chars = list(json.load(stream)["kept_characteristics"])
        else:
            sample = pd.read_parquet(next(iter(self.files.values())))
            chars = [column for column in sample.columns if column not in META_COLUMNS]
        if self.add_log_me:
            chars.append("log_me_rank")
        return chars

    @property
    def input_dim(self) -> int:
        return len(self.characteristics)

    def _build_monthly_state(self) -> pd.DataFrame:
        pieces = []
        columns = ["eom", "me", "ret_exc_lead1m"]
        for year, path in self.files.items():
            frame = pd.read_parquet(path, columns=columns)
            frame["eom"] = pd.to_datetime(frame["eom"])
            # Keep the investable universe defined at formation time. Missing
            # next-month returns are an outcome-data issue and must not decide
            # which stocks existed in today's cross-section.
            frame["ret_filled"] = frame["ret_exc_lead1m"].fillna(0.0)
            frame["mr"] = frame["me"] * frame["ret_filled"]
            grouped = frame.groupby("eom", sort=True)
            stats = grouped.agg(
                ew_fwd=("ret_filled", "mean"),
                xs_disp_fwd=("ret_filled", "std"),
                me_sum=("me", "sum"),
                mr_sum=("mr", "sum"),
                n_stocks=("ret_filled", "size"),
            )
            stats["year"] = year
            pieces.append(stats)
        state = pd.concat(pieces).sort_index()
        state["vw_fwd"] = state["mr_sum"] / state["me_sum"].replace(0, np.nan)
        ew = state["ew_fwd"]
        state["ew_lag1"] = ew.shift(1)
        state["vw_lag1"] = state["vw_fwd"].shift(1)
        state["ew_mean12"] = ew.shift(1).rolling(12, min_periods=6).mean()
        state["ew_vol12"] = ew.shift(1).rolling(12, min_periods=6).std()
        state["xs_disp_lag1"] = state["xs_disp_fwd"].shift(1)
        state["log_n"] = np.log(state["n_stocks"].clip(lower=1))
        return state

    @property
    def state_columns(self) -> list[str]:
        return [
            "ew_lag1", "vw_lag1", "ew_mean12", "ew_vol12",
            "xs_disp_lag1", "log_n",
        ]

    def fit_state_scaler(self, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
        block = self.monthly_state.loc[pd.Timestamp(start):pd.Timestamp(end), self.state_columns]
        mean = block.mean().to_numpy(dtype=np.float32)
        std = block.std().replace(0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
        return mean, std

    def dates(self, start: str, end: str) -> list[pd.Timestamp]:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return [
            date for date in self.monthly_state.index
            if start_ts <= date <= end_ts and date.year in self.files
        ]

    def years(self, start: str, end: str) -> list[int]:
        return sorted({date.year for date in self.dates(start, end)})

    def _state_for(self, date: pd.Timestamp, scaler) -> np.ndarray:
        values = self.monthly_state.loc[date, self.state_columns].to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = scaler
        return ((values - mean) / std).astype(np.float32)

    def _load_year(self, year: int) -> pd.DataFrame:
        base_chars = [c for c in self.characteristics if c != "log_me_rank"]
        columns = ["id", "eom", "me", "ret_exc_lead1m", *base_chars]
        frame = pd.read_parquet(self.files[year], columns=columns)
        frame["eom"] = pd.to_datetime(frame["eom"])
        # Do not condition today's universe on availability of tomorrow's return.
        frame = frame.copy()
        if self.add_log_me:
            log_me = np.log(frame["me"].astype(float).clip(lower=1e-12))
            ranks = log_me.groupby(frame["eom"]).rank(method="average", pct=True)
            frame["log_me_rank"] = ranks.astype(np.float32) - 0.5
        return frame

    def iter_year(
        self,
        year: int,
        start: str,
        end: str,
        scaler: tuple[np.ndarray, np.ndarray],
    ):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frame = self._load_year(year)
        frame = frame.loc[frame["eom"].between(start_ts, end_ts)]
        for date, month in frame.groupby("eom", sort=True):
            x = month[self.characteristics].to_numpy(dtype=np.float32, copy=True)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            raw_r = month["ret_exc_lead1m"].to_numpy(dtype=np.float32)
            observed = np.isfinite(raw_r)
            # Missing payoff observations are assigned zero in the baseline. This
            # is conservative and, unlike dropping them, does not use future
            # availability to alter formation-time weights. Sensitivity to this
            # convention is reported separately.
            r = np.nan_to_num(raw_r, nan=0.0, posinf=0.0, neginf=0.0)
            yield MonthPanel(
                date=pd.Timestamp(date),
                x=x,
                r=r,
                ids=month["id"].to_numpy(copy=True),
                me=month["me"].to_numpy(dtype=np.float32, copy=True),
                state=self._state_for(pd.Timestamp(date), scaler),
                return_observed=observed,
            )

    def materialize(
        self,
        start: str,
        end: str,
        scaler: tuple[np.ndarray, np.ndarray],
        *,
        cache: bool = True,
    ) -> list[MonthPanel]:
        """Materialize a date range once; characteristics are stored as float16."""
        mean, std = scaler
        scaler_key = tuple(np.round(np.r_[mean, std], 6).tolist())
        key = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), scaler_key)
        if cache and key in self._panel_cache:
            return self._panel_cache[key]
        panels: list[MonthPanel] = []
        for year in self.years(start, end):
            for panel in self.iter_year(year, start, end, scaler):
                panel.x = panel.x.astype(np.float16, copy=False)
                panels.append(panel)
        if cache:
            self._panel_cache[key] = panels
        return panels
