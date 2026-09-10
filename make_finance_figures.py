from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_factors.benchmarks import benchmark_model_columns, tangency_weights, oos_hedged_factor
from neural_factors.train import annualized_sharpe

ROOT = Path(__file__).resolve().parent
REP = ROOT / "results/causal_v2_k_grid_paper/k32_seed04"
GRID = ROOT / "results/causal_v2_k_grid_paper"
LINEAR = ROOT / "results/causal_v2_linear_managed"
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.dpi": 140, "savefig.dpi": 300})

def clean(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    return ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def scale_to_vol(x, target=0.10):
    x = pd.Series(x, dtype=float)
    vol = np.sqrt(12.0) * x.std(ddof=1)
    return x * (target / vol if np.isfinite(vol) and vol > 0 else 1.0)


def wealth(x):
    return (1.0 + pd.Series(x, dtype=float).fillna(0.0)).cumprod()


def rolling_sharpe(x, window=36):
    x = pd.Series(x, dtype=float)
    return np.sqrt(12.0) * x.rolling(window).mean() / x.rolling(window).std(ddof=1)


def drawdown(x):
    w = wealth(x)
    return w / w.cummax() - 1.0

def load_core():
    ens = pd.read_parquet(GRID / "seed_ensemble_returns.parquet")
    ens = ens.loc[ens["kmax"] == 32, ["payoff_date", "ensemble_return"]].copy()
    ens["payoff_date"] = pd.to_datetime(ens["payoff_date"])
    linear = pd.read_parquet(LINEAR / "test_returns.parquet").copy()
    linear["payoff_date"] = pd.to_datetime(linear["payoff_date"])
    rep = pd.read_parquet(REP / "monthly_test.parquet").copy()
    rep["payoff_date"] = pd.to_datetime(rep["payoff_date"])
    bench = pd.read_parquet(REP / "spanning/benchmark_panel.parquet").copy()
    bench["date"] = pd.to_datetime(bench["date"])
    return ens, linear, rep, bench


def benchmark_tangency(bench):
    estimation = bench.loc[bench["date"] <= pd.Timestamp("2014-12-31")].copy()
    test = bench.loc[bench["date"].between("2015-01-31", "2024-12-31")].copy()
    out = pd.DataFrame({"payoff_date": test["date"].to_numpy()})
    for name in ("FF5+MOM", "QMJ", "Mispricing-JKP", "JKP-core"):
        cols = benchmark_model_columns()[name]
        w = tangency_weights(estimation, cols, ridge=1e-3)
        block = test[cols]
        out[name] = block.to_numpy(dtype=float) @ w
    return out


def common_panel():
    ens, linear, rep, bench = load_core()
    p = ens.merge(linear[["payoff_date", "portfolio_return"]], on="payoff_date", how="inner")
    p = p.rename(columns={"ensemble_return": "Neural ensemble", "portfolio_return": "Linear characteristics"})
    p = p.merge(benchmark_tangency(bench), on="payoff_date", how="inner")
    return p, rep, bench

def figure_equity_flagship(panel):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for col in ["Neural ensemble", "Linear characteristics", "FF5+MOM", "JKP-core"]:
        x = scale_to_vol(panel[col])
        ax.plot(panel["payoff_date"], wealth(x), label=col, linewidth=1.8)
    ax.set_ylabel("Growth of $1")
    ax.set_title("Out-of-sample cumulative performance, 2015-2024")
    ax.legend(frameon=False, ncol=2)
    clean(ax)
    save(fig, "equity_flagship")


def figure_rolling_sharpe(panel):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for col in ["Neural ensemble", "Linear characteristics", "FF5+MOM", "JKP-core"]:
        ax.plot(panel["payoff_date"], rolling_sharpe(panel[col]), label=col, linewidth=1.6)
    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.set_ylabel("36-month annualized Sharpe")
    ax.set_title("Rolling out-of-sample performance")
    ax.legend(frameon=False, ncol=2)
    clean(ax)
    save(fig, "rolling_sharpe")


def figure_drawdown(panel):
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    for col in ["Neural ensemble", "Linear characteristics", "JKP-core"]:
        ax.plot(panel["payoff_date"], drawdown(scale_to_vol(panel[col])), label=col, linewidth=1.5)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdowns at a common 10% annualized volatility")
    ax.legend(frameon=False)
    clean(ax)
    save(fig, "drawdown")

def figure_cost_equity():
    c = pd.read_csv(REP / "investability/monthly_costs.csv")
    c["payoff_date"] = pd.to_datetime(c["payoff_date"])
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    specs = [("vol_target_return", "Gross"), ("net_return_5bps", "5 bps"),
             ("net_return_10bps", "10 bps"), ("net_return_25bps", "25 bps")]
    for col, label in specs:
        ax.plot(c["payoff_date"], wealth(c[col]), label=label, linewidth=1.7)
    ax.set_ylabel("Growth of $1")
    ax.set_title("Implementable equity lines after trading costs")
    ax.legend(frameon=False, ncol=2)
    clean(ax)
    save(fig, "equity_costs")


def figure_cost_sensitivity():
    s = pd.read_csv(REP / "investability/summary.csv")
    order = ["vol_target_return", "net_return_5bps", "net_return_10bps", "net_return_25bps",
             "net_return_10bps_borrow100bps", "net_return_10bps_borrow300bps"]
    labels = ["Gross", "5 bps", "10 bps", "25 bps", "10 bps + 100 bps borrow", "10 bps + 300 bps borrow"]
    vals = [float(s.loc[s.series == x, "annualized_sharpe"].iloc[0]) for x in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)), labels, rotation=24, ha="right")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Sensitivity to implementation costs")
    clean(ax)
    save(fig, "cost_sensitivity")

def figure_k_capacity():
    g = pd.read_csv(GRID / "robustness_summary.csv")
    e = pd.read_csv(GRID / "seed_ensemble_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax = axes[0]
    ax.errorbar(g.kmax, g.test_sharpe_mean, yerr=g.test_sharpe_std, marker="o", capsize=3, label="Mean +/- 1 SD")
    ax.plot(e.kmax, e.ensemble_sharpe, marker="s", label="10-seed ensemble")
    ax.set_xscale("log", base=2); ax.set_xticks(g.kmax, g.kmax.astype(int))
    ax.set_xlabel("Maximum factor capacity $K_{max}$"); ax.set_ylabel("OOS Sharpe")
    ax.set_title("Capacity improves portfolio performance"); ax.legend(frameon=False); clean(ax)
    ax = axes[1]
    ax.plot(g.kmax, g.allocation_rank_mean, marker="o", label="Allocation rank")
    ax.plot(g.kmax, g.first_direction_share_mean, marker="s", label="First-direction share")
    ax.set_xscale("log", base=2); ax.set_xticks(g.kmax, g.kmax.astype(int))
    ax.set_xlabel("Maximum factor capacity $K_{max}$")
    ax.set_title("Economic use remains concentrated"); ax.legend(frameon=False); clean(ax)
    save(fig, "k_capacity")


def figure_seed_dispersion():
    g = pd.read_csv(GRID / "robustness_grid.csv")
    groups = [g.loc[g.kmax == k, "test_sharpe"].to_numpy() for k in sorted(g.kmax.unique())]
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.boxplot(groups, tick_labels=[str(int(k)) for k in sorted(g.kmax.unique())], showfliers=False)
    ax.set_xlabel("Maximum factor capacity $K_{max}$"); ax.set_ylabel("OOS Sharpe")
    ax.set_title("Performance across random initializations")
    clean(ax)
    save(fig, "seed_dispersion")

def figure_marginal_sharpe():
    m = pd.read_csv(REP / "spanning/marginal_sharpe.csv")
    x = np.arange(len(m)); width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - width/2, m.benchmark_sharpe, width, label="Benchmark")
    ax.bar(x + width/2, m.augmented_sharpe, width, label="Benchmark + learned factor")
    ax.set_xticks(x, m.model, rotation=18, ha="right")
    ax.set_ylabel("OOS Sharpe"); ax.set_title("Marginal value of the learned factor")
    ax.legend(frameon=False); clean(ax)
    save(fig, "marginal_sharpe")


def figure_spanning_alpha():
    s = pd.read_csv(REP / "spanning/spanning_alpha.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    axes[0].bar(s.model, 100*s.alpha_annualized)
    axes[0].set_ylabel("Annualized alpha (%)"); axes[0].set_title("Spanning alpha")
    axes[0].tick_params(axis="x", rotation=20); clean(axes[0])
    axes[1].bar(s.model, s.alpha_t_nw)
    axes[1].axhline(1.96, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel("Newey-West $t$-statistic"); axes[1].set_title("Statistical strength")
    axes[1].tick_params(axis="x", rotation=20); clean(axes[1])
    save(fig, "spanning_alpha")

def load_scaled_canonical_and_benchmarks():
    frames = []
    for split in ("train", "val", "test"):
        f = pd.read_parquet(REP / f"factors/canonical_{split}.parquet")[["date", "canonical_factor_00"]]
        f = f.rename(columns={"canonical_factor_00": "learned_factor"})
        f["date"] = pd.to_datetime(f["date"]) + pd.offsets.MonthEnd(1)
        f["split"] = split; frames.append(f)
    allf = pd.concat(frames, ignore_index=True)
    est = allf.loc[allf.split.isin(["train", "val"]), "learned_factor"]
    scale = 0.10 / max(np.sqrt(12.0)*est.std(ddof=1), 1e-12)
    allf["learned_factor"] *= scale
    bench = pd.read_parquet(REP / "spanning/benchmark_panel.parquet")
    bench["date"] = pd.to_datetime(bench["date"])
    merged = allf.merge(bench, on="date", how="inner")
    return merged.loc[merged.split.isin(["train", "val"])], merged.loc[merged.split == "test"]


def figure_residual_equity():
    est, test = load_scaled_canonical_and_benchmarks()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    raw = test[["date", "learned_factor"]].dropna()
    ax.plot(raw.date, wealth(raw.learned_factor), label="Raw learned factor", linewidth=1.8)
    for name in ("FF5+MOM", "JKP-core"):
        h = oos_hedged_factor(est, test, "learned_factor", benchmark_model_columns()[name])
        ax.plot(h.date, wealth(h.hedged_factor), label=f"Hedged vs {name}", linewidth=1.6)
    ax.set_ylabel("Growth of $1"); ax.set_title("Benchmark-neutralized learned factor")
    ax.legend(frameon=False); clean(ax)
    save(fig, "residual_equity")

def figure_policy_spectrum():
    p = pd.read_csv(REP / "policy/policy_spectrum.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    show = p.head(10)
    axes[0].bar(show.direction, show.share)
    axes[0].set_xlabel("Policy direction"); axes[0].set_ylabel("Share")
    axes[0].set_title("Policy spectrum"); clean(axes[0])
    axes[1].plot(p.direction, p.cumulative_share, marker="o", markersize=3)
    axes[1].axhline(0.95, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].axhline(0.99, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xlim(1, min(12, len(p))); axes[1].set_ylim(0.8, 1.005)
    axes[1].set_xlabel("Number of directions"); axes[1].set_ylabel("Cumulative share")
    axes[1].set_title("Rapid concentration of policy capacity"); clean(axes[1])
    save(fig, "policy_spectrum")


def figure_truncation():
    t = pd.read_csv(REP / "factors/canonical_test_truncation.csv")
    show = t.loc[t["rank"] <= 10]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].plot(show["rank"], show["sharpe"], marker="o")
    axes[0].axhline(t.iloc[-1]["sharpe"], linestyle="--", linewidth=0.9, alpha=0.6, label="Full portfolio")
    axes[0].set_xlabel("Canonical directions retained"); axes[0].set_ylabel("OOS Sharpe")
    axes[0].set_title("Economic value under truncation"); axes[0].legend(frameon=False); clean(axes[0])
    axes[1].plot(show["rank"], 100*show["relative_payoff_mse"], marker="o")
    axes[1].set_yscale("log"); axes[1].set_xlabel("Canonical directions retained")
    axes[1].set_ylabel("Relative payoff MSE (%)"); axes[1].set_title("Reconstruction error")
    clean(axes[1])
    save(fig, "truncation")

PRETTY = {
    "ret_12_1": "Momentum (12-1)", "resff3_12_1": "Residual momentum",
    "rmax5_21d": "Max return (5d)", "rmax1_21d": "Max return (1d)",
    "mispricing_perf": "Mispricing performance", "fcf_me": "Free cash flow / ME",
    "rvol_21d": "Realized volatility", "ivol_capm_21d": "Idio. vol. CAPM",
    "ivol_hxz4_21d": "Idio. vol. q-factor", "ivol_ff3_21d": "Idio. vol. FF3",
    "ret_9_1": "Momentum (9-1)", "ebit_bev": "EBIT / enterprise value",
    "qmj_safety": "QMJ safety", "rmax5_rvol_21d": "Extreme return / volatility",
    "ret_1_0": "One-month return", "iskew_ff3_21d": "Idiosyncratic skewness",
    "o_score": "O-score", "dgp_dsale": "Gross-margin / sales change",
    "capx_gr1": "Investment growth", "dolvol_126d": "Dollar volume",
    "ami_126d": "Amihud illiquidity", "log_me_rank": "Size",
}

def pretty(x):
    return PRETTY.get(str(x), str(x).replace("_", " "))


def figure_factor_profiles():
    f = pd.read_csv(REP / "interpretation/factor_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.5))
    for ax, factor in zip(axes, (1, 2)):
        d = f.loc[f.factor == factor].nlargest(10, "abs_top_minus_bottom").sort_values("top_minus_bottom")
        ax.barh([pretty(x) for x in d.characteristic], d.top_minus_bottom)
        ax.axvline(0, linewidth=0.8, alpha=0.5); ax.set_xlabel("Top-minus-bottom characteristic rank")
        ax.set_title(f"Canonical factor {factor}"); clean(ax, grid=False)
    save(fig, "factor_profiles")

def figure_interactions():
    d = pd.read_csv(REP / "interpretation/factor_interactions.csv")
    d = d.loc[d.factor == 1].head(12).iloc[::-1]
    labels = [f"{pretty(a)} × {pretty(b)}" for a, b in zip(d.characteristic_1, d.characteristic_2)]
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    ax.barh(labels, d.normalized_interaction)
    ax.set_xlabel("Normalized nonlinear interaction")
    ax.set_title("Strongest interactions in the leading learned factor")
    clean(ax)
    save(fig, "interactions")


def figure_interaction_stability():
    s = pd.read_csv(REP / "stability/interaction_stability.csv")
    names = ["train early", "train late", "validation", "test"]
    keys = ["train_early", "train_late", "validation", "test"]
    M = np.eye(4)
    for row in s.itertuples():
        i, j = keys.index(row.period_1), keys.index(row.period_2)
        M[i, j] = M[j, i] = row.spearman_interaction_rank
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(4), names, rotation=25, ha="right"); ax.set_yticks(range(4), names)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="white" if M[i,j] < 0.9 else "black")
    ax.set_title("Stability of learned interaction rankings")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rank correlation")
    save(fig, "interaction_stability")

def figure_state_dependence():
    s = pd.read_csv(REP / "interpretation/allocator_state_links.csv")
    labels_map = {"ew_lag1":"EW market, lag 1", "vw_lag1":"VW market, lag 1",
                  "ew_mean12":"12m market trend", "ew_vol12":"12m market vol",
                  "xs_disp_lag1":"Cross-sectional dispersion", "log_n":"Log number of stocks"}
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.8))
    for ax, factor in zip(axes, (1,2)):
        d = s.loc[s.factor == factor].sort_values("correlation")
        ax.barh([labels_map.get(x,x) for x in d.state], d.correlation)
        ax.axvline(0, linewidth=0.8, alpha=0.5); ax.set_xlim(-0.8,0.8)
        ax.set_title(f"Factor {factor} allocation"); ax.set_xlabel("Correlation with state variable")
        clean(ax, grid=False)
    save(fig, "state_dependence")


def figure_annual_returns(panel):
    p = panel.copy(); p["year"] = p.payoff_date.dt.year
    cols = ["Neural ensemble", "Linear characteristics", "JKP-core"]
    scaled = p[["payoff_date","year"]].copy()
    for c in cols: scaled[c] = scale_to_vol(p[c]).to_numpy()
    annual = scaled.groupby("year")[cols].apply(lambda x: (1+x).prod()-1)
    x = np.arange(len(annual)); width = 0.26
    fig, ax = plt.subplots(figsize=(8.1, 4.0))
    for j,c in enumerate(cols): ax.bar(x+(j-1)*width, 100*annual[c], width, label=c)
    ax.set_xticks(x, annual.index.astype(str), rotation=35); ax.set_ylabel("Annual return (%)")
    ax.set_title("Year-by-year out-of-sample performance"); ax.legend(frameon=False, ncol=3)
    ax.axhline(0, linewidth=0.8, alpha=0.5); clean(ax)
    save(fig, "annual_returns")

def figure_turnover_exposure():
    c = pd.read_csv(REP / "investability/monthly_costs.csv")
    c["payoff_date"] = pd.to_datetime(c["payoff_date"])
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
    axes[0].plot(c.payoff_date, c.turnover_half_l1, linewidth=1.3)
    axes[0].axhline(c.turnover_half_l1.mean(), linestyle="--", linewidth=0.9,
                    label=f"Mean = {c.turnover_half_l1.mean():.2f}")
    axes[0].set_ylabel("Half-L1 turnover"); axes[0].legend(frameon=False); clean(axes[0])
    axes[1].plot(c.payoff_date, c.gross_exposure, label="Gross exposure", linewidth=1.3)
    axes[1].plot(c.payoff_date, c.short_exposure, label="Short exposure", linewidth=1.3)
    axes[1].set_ylabel("Exposure"); axes[1].legend(frameon=False); clean(axes[1])
    axes[0].set_title("Turnover and exposure over time")
    save(fig, "turnover_exposure")

def figure_factor_equity():
    c = pd.read_parquet(REP / "factors/canonical_test.parquet")
    c["payoff_date"] = pd.to_datetime(c["date"]) + pd.offsets.MonthEnd(1)
    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    for j in (0, 1, 2):
        col = f"canonical_factor_{j:02d}"
        if col in c.columns:
            ax.plot(c.payoff_date, wealth(scale_to_vol(c[col])),
                    label=f"Canonical factor {j+1}", linewidth=1.6)
    ax.set_ylabel("Growth of $1")
    ax.set_title("Equity lines of the leading canonical factors")
    ax.legend(frameon=False); clean(ax)
    save(fig, "factor_equity")

def figure_dimension_by_period():
    d = pd.read_csv(REP / "stability/allocation_dimension_by_period.csv")
    labels = ["Train early", "Train late", "Validation", "Test"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.bar(x, d.entropy_rank, label="Allocation effective rank")
    ax.plot(x, d.first_direction_share, marker="o", label="First-direction share")
    ax.set_xticks(x, labels); ax.set_ylabel("Rank / share")
    ax.set_title("Economic concentration across subperiods")
    ax.legend(frameon=False); clean(ax)
    save(fig, "dimension_by_period")


def figure_return_distribution(panel):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for col in ["Neural ensemble", "Linear characteristics", "JKP-core"]:
        x = scale_to_vol(panel[col])
        ax.hist(x, bins=18, density=True, histtype="step", linewidth=1.5, label=col)
    ax.axvline(0, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Monthly return at 10% annualized volatility")
    ax.set_ylabel("Density"); ax.set_title("Distribution of out-of-sample returns")
    ax.legend(frameon=False); clean(ax)
    save(fig, "return_distribution")

def figure_ablations():
    path = ROOT / "results/causal_v2_ablations_paper/ablation_summary.csv"
    if not path.exists():
        return
    a = pd.read_csv(path)
    g = a.groupby("model", sort=False).agg(test_mean=("test_sharpe","mean"), test_sd=("test_sharpe","std")).reset_index()
    order=["full","no_state","no_context","static","linear_characteristics","direct_conditional_nn"]
    labels={"full":"Full factor model","no_state":"No aggregate state","no_context":"No cross-sectional pooling",
            "static":"Static allocation","linear_characteristics":"Linear characteristic map",
            "direct_conditional_nn":"Direct conditional NN"}
    g=g.set_index("model").loc[order].reset_index()
    fig, ax=plt.subplots(figsize=(7.8,4.0))
    x=np.arange(len(g))
    ax.bar(x,g.test_mean,yerr=g.test_sd,capsize=3)
    ax.set_xticks(x,[labels[m] for m in g.model],rotation=24,ha="right")
    ax.set_ylabel("Mean OOS Sharpe")
    ax.set_title("Economic architecture ablations; five seeds each")
    clean(ax)
    save(fig,"ablations")

def figure_walk_forward(panel):
    root = ROOT / "results/causal_v2_walkforward_ensemble5"
    if not (root / "walk_forward_ensemble_returns.parquet").exists():
        return
    w = pd.read_parquet(root / "walk_forward_ensemble_returns.parquet")
    w["payoff_date"] = pd.to_datetime(w["payoff_date"])
    annual = pd.read_csv(root / "annual_summary.csv")
    fixed = panel[["payoff_date", "Neural ensemble"]].copy()
    linear_path = ROOT / "results/causal_v2_linear_walkforward/walk_forward_returns.parquet"
    linear = pd.read_parquet(linear_path)[["payoff_date","portfolio_return"]].rename(columns={"portfolio_return":"Recursive linear"})
    linear["payoff_date"] = pd.to_datetime(linear["payoff_date"])
    merged = w[["payoff_date", "ensemble_return"]].merge(fixed, on="payoff_date", how="inner").merge(linear,on="payoff_date",how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    for col, label in [("ensemble_return", "Recursive neural ensemble"),
                       ("Neural ensemble", "Fixed-split neural ensemble"),
                       ("Recursive linear", "Recursive linear characteristics")]:
        axes[0].plot(merged.payoff_date, wealth(scale_to_vol(merged[col])), label=label, linewidth=1.6)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Expanding-window OOS performance")
    axes[0].legend(frameon=False, fontsize=8)
    clean(axes[0])
    x=np.arange(len(annual)); width=0.36
    y=annual.ensemble_sharpe.to_numpy()
    lin_fold = pd.read_csv(ROOT / "results/causal_v2_linear_walkforward/walk_forward_folds.csv").set_index("payoff_year")
    ly=np.array([lin_fold.loc[int(year),"sharpe"] for year in annual.year])
    axes[1].bar(x-width/2,y,width,label="Recursive neural ensemble")
    axes[1].bar(x+width/2,ly,width,label="Recursive linear")
    axes[1].vlines(x-width/2, annual.seed_min.to_numpy(), annual.seed_max.to_numpy(), linewidth=1.1, label="Neural seed range")
    axes[1].axhline(0, linewidth=0.8, alpha=0.5)
    axes[1].set_xticks(x, annual.year.astype(int).astype(str), rotation=35, ha="right")
    axes[1].set_ylabel("Annualized Sharpe")
    axes[1].set_title("Payoff-year Sharpe")
    axes[1].legend(frameon=False, fontsize=7)
    clean(axes[1])
    save(fig, "walk_forward")

def main():
    panel, rep, bench = common_panel()
    figure_equity_flagship(panel)
    figure_rolling_sharpe(panel)
    figure_drawdown(panel)
    figure_cost_equity()
    figure_cost_sensitivity()
    figure_k_capacity()
    figure_seed_dispersion()
    figure_marginal_sharpe()
    figure_spanning_alpha()
    figure_residual_equity()
    figure_policy_spectrum()
    figure_truncation()
    figure_factor_profiles()
    figure_interactions()
    figure_interaction_stability()
    figure_state_dependence()
    figure_annual_returns(panel)
    figure_factor_equity()
    figure_dimension_by_period()
    figure_return_distribution(panel)
    figure_turnover_exposure()
    figure_ablations()
    figure_walk_forward(panel)
    print(f"Wrote finance figure suite to {FIG}")


if __name__ == "__main__":
    main()
