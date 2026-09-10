from __future__ import annotations

from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


def month_end(values) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp("M")


def payoff_month_from_formation(values) -> pd.Series:
    """Map end-of-month formation date t to realized-return month t+1."""
    dates = pd.to_datetime(values)
    return (dates.dt.to_period("M") + 1).dt.to_timestamp("M")


def download_ff6(start="1963-01-01", end="2024-12-31") -> pd.DataFrame:
    import tidyfinance as tf
    ff5 = tf.download_data(
        domain="Fama-French", dataset="factors_ff_5_2x3_monthly",
        start_date=start, end_date=end,
    )
    mom = tf.download_data(
        domain="Fama-French", dataset="factors_ff_momentum_factor_monthly",
        start_date=start, end_date=end,
    )
    ff5["date"] = month_end(ff5["date"])
    mom["date"] = month_end(mom["date"])
    return ff5.merge(mom, on="date", how="inner")


def download_q5(start="1967-01-01", end="2024-12-31") -> pd.DataFrame:
    import tidyfinance as tf
    q5 = tf.download_data(
        domain="Global Q", dataset="q5_factors_monthly",
        start_date=start, end_date=end,
    )
    q5["date"] = month_end(q5["date"])
    return q5


def download_aqr_qmj(start="1963-01-01", end="2024-12-31") -> pd.DataFrame:
    url = (
        "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
        "Quality-Minus-Junk-Factors-Monthly.xlsx"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    qmj = pd.read_excel(io.BytesIO(response.content), sheet_name="QMJ Factors", header=18)
    out = qmj[["DATE", "USA"]].rename(columns={"DATE": "date", "USA": "qmj"})
    out["date"] = month_end(out["date"])
    out["qmj"] = pd.to_numeric(out["qmj"], errors="coerce")
    mask = out["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return out.loc[mask].dropna().reset_index(drop=True)


def download_jkp_factors(
    names: list[str] | tuple[str, ...],
    start="1963-01-01",
    end="2024-12-31",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    url = (
        "https://jkpfactors-data.s3.amazonaws.com/public/"
        "%5Busa%5D_%5Ball_factors%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip"
    )
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "jkp_usa_all_factors_monthly_vw_cap.zip"
    if cache_path is not None and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        content = response.content
        if cache_path is not None:
            cache_path.write_bytes(content)
    archive = zipfile.ZipFile(io.BytesIO(content))
    frame = pd.read_csv(archive.open(archive.namelist()[0]))
    frame = frame.loc[frame["name"].isin(list(names)), ["date", "name", "ret"]].copy()
    frame["date"] = month_end(frame["date"])
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    wide = frame.pivot(index="date", columns="name", values="ret").reset_index()
    mask = wide["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return wide.loc[mask].sort_values("date").reset_index(drop=True)


def download_stambaugh_yuan(start="1963-01-01", end="2016-12-31") -> pd.DataFrame:
    url = "https://finance.wharton.upenn.edu/~stambaug/M4.csv"
    frame = pd.read_csv(url)
    year = (frame["YYYYMM"] // 100).astype(int)
    month = (frame["YYYYMM"] % 100).astype(int)
    frame["date"] = pd.to_datetime(dict(year=year, month=month, day=1)).dt.to_period("M").dt.to_timestamp("M")
    frame = frame.rename(columns={
        "MKTRF": "sy_mkt_excess", "SMB": "sy_smb", "MGMT": "sy_mgmt",
        "PERF": "sy_perf", "RF": "sy_rf",
    })
    mask = frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return frame.loc[mask, ["date", "sy_mkt_excess", "sy_smb", "sy_mgmt", "sy_perf", "sy_rf"]].reset_index(drop=True)


def default_benchmark_panel(
    start="1967-01-01",
    end="2024-12-31",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    ff = download_ff6(start, end)
    q5 = download_q5(start, end)
    qmj = download_aqr_qmj(start, end)
    jkp_names = [
        "be_me", "ret_12_1", "gp_at", "at_gr1", "cop_at",
        "mispricing_perf", "mispricing_mgmt", "qmj_prof", "resff3_12_1",
    ]
    jkp = download_jkp_factors(jkp_names, start, end, cache_dir=cache_dir)
    out = ff.merge(q5, on="date", how="outer", suffixes=("_ff", "_q"))
    out = out.merge(qmj, on="date", how="outer").merge(jkp, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def nw_spanning(
    frame: pd.DataFrame,
    target: str,
    regressors: list[str],
    lags: int = 6,
) -> dict[str, float]:
    data = frame[[target, *regressors]].dropna()
    y = data[target].to_numpy(dtype=float)
    x = sm.add_constant(data[regressors].to_numpy(dtype=float))
    fit = sm.OLS(y, x).fit()
    hac = cov_hac(fit, nlags=lags)
    se = np.sqrt(np.diag(hac))
    alpha = float(fit.params[0])
    return {
        "months": int(len(data)),
        "alpha_monthly": alpha,
        "alpha_annualized": 12.0 * alpha,
        "alpha_t_nw": float(alpha / se[0]) if se[0] > 0 else np.nan,
        "r2": float(fit.rsquared),
    }


def tangency_weights(frame: pd.DataFrame, columns: list[str], ridge: float = 1e-3) -> np.ndarray:
    data = frame[columns].dropna().to_numpy(dtype=float)
    mu = data.mean(axis=0)
    cov = np.cov(data, rowvar=False, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    scale = np.trace(cov) / max(len(columns), 1)
    regularized = cov + ridge * max(float(scale), 1e-8) * np.eye(len(columns))
    return np.linalg.pinv(regularized) @ mu


def realized_span_sharpe(frame: pd.DataFrame, columns: list[str], weights: np.ndarray) -> float:
    data = frame[columns].dropna()
    values = data.to_numpy(dtype=float) @ weights
    if len(values) < 2 or values.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(12.0) * values.mean() / values.std(ddof=1))


def oos_marginal_sharpe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    learned: str,
    benchmarks: list[str],
    ridge: float = 1e-3,
) -> dict[str, float]:
    bench_w = tangency_weights(train, benchmarks, ridge=ridge)
    augmented_cols = [*benchmarks, learned]
    aug_w = tangency_weights(train, augmented_cols, ridge=ridge)
    bench_sr = realized_span_sharpe(test, benchmarks, bench_w)
    aug_sr = realized_span_sharpe(test, augmented_cols, aug_w)
    learned_sr = realized_span_sharpe(test, [learned], np.array([1.0]))
    return {
        "benchmark_sharpe": bench_sr,
        "augmented_sharpe": aug_sr,
        "marginal_sharpe": float(aug_sr - bench_sr),
        "learned_factor_sharpe": learned_sr,
    }


def benchmark_model_columns() -> dict[str, list[str]]:
    return {
        "FF5+MOM": ["mkt_excess_ff", "smb", "hml", "rmw", "cma", "mom"],
        "q5": ["mkt_excess_q", "me", "ia", "roe", "eg"],
        "QMJ": ["mkt_excess_ff", "smb", "qmj"],
        "Mispricing-JKP": ["mkt_excess_ff", "smb", "mispricing_mgmt", "mispricing_perf"],
        "JKP-core": [
            "mkt_excess_ff", "smb", "be_me", "ret_12_1", "gp_at", "at_gr1",
            "cop_at", "resff3_12_1", "qmj_prof", "mispricing_mgmt", "mispricing_perf",
        ],
    }


def oos_hedged_factor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    learned: str,
    benchmarks: list[str],
) -> pd.DataFrame:
    fit_data = train[[learned, *benchmarks]].dropna()
    x_train = sm.add_constant(fit_data[benchmarks].to_numpy(dtype=float))
    fit = sm.OLS(fit_data[learned].to_numpy(dtype=float), x_train).fit()
    beta = fit.params[1:]
    test_data = test[["date", learned, *benchmarks]].dropna().copy()
    hedge = test_data[benchmarks].to_numpy(dtype=float) @ beta
    test_data["hedged_factor"] = test_data[learned].to_numpy(dtype=float) - hedge
    test_data["raw_factor"] = test_data[learned]
    return test_data[["date", "raw_factor", "hedged_factor"]]
