from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build LaTeX tables, macros, and paper figures")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--robustness", default="results/robustness_grid_v3")
    parser.add_argument("--ablations", default="results/ablations_v3")
    parser.add_argument("--walk-forward", default="results/walk_forward_v3")
    parser.add_argument("--paper-dir", default="paper")
    return parser.parse_args()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def pct(value):
    return f"{100.0 * value:.1f}\\%"


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_macros(baseline: Path, robustness: Path, walk: Path, paper: Path):
    metrics = read_json(baseline / "metrics.json")
    spectrum = pd.read_csv(baseline / "factors" / "economic_factor_spectrum.csv")
    spanning = pd.read_csv(baseline / "spanning" / "spanning_alpha.csv")
    marginal = pd.read_csv(baseline / "spanning" / "marginal_sharpe.csv")
    costs = pd.read_csv(baseline / "investability" / "summary.csv")

    walk_sr = np.nan
    walk_summary = walk / "walk_forward_summary.json"
    if walk_summary.exists():
        walk_sr = read_json(walk_summary)["annualized_sharpe"]

    entropy = read_json(baseline / "factors" / "economic_dimension.json")["entropy_rank"]
    ff_t = spanning.loc[spanning["model"] == "FF5+MOM", "alpha_t_nw"].iloc[0]
    jkp_t = spanning.loc[spanning["model"] == "JKP-core", "alpha_t_nw"].iloc[0]
    marginal_jkp = marginal.loc[marginal["model"] == "JKP-core", "marginal_sharpe"].iloc[0]
    cost_row = costs.loc[costs["series"] == "net_return_10bps"]
    net_sr = float(cost_row["annualized_sharpe"].iloc[0]) if len(cost_row) else np.nan

    text = "\n".join([
        f"\\newcommand{{\\BaselineTestSharpe}}{{{metrics['test']['sharpe']:.2f}}}",
        f"\\newcommand{{\\WalkForwardSharpe}}{{{walk_sr:.2f}}}",
        f"\\newcommand{{\\KEff}}{{{entropy:.2f}}}",
        f"\\newcommand{{\\FirstShare}}{{{pct(float(spectrum.loc[0, 'share']))}}}",
        f"\\newcommand{{\\FFAlphaT}}{{{ff_t:.2f}}}",
        f"\\newcommand{{\\JKPAlphaT}}{{{jkp_t:.2f}}}",
        f"\\newcommand{{\\MarginalSharpe}}{{{marginal_jkp:.2f}}}",
        f"\\newcommand{{\\NetSharpeTen}}{{{net_sr:.2f}}}",
    ]) + "\n"
    write(paper / "tables" / "results_macros.tex", text)


def baseline_table(baseline: Path, walk: Path, paper: Path):
    metrics = read_json(baseline / "metrics.json")
    walk_sr = np.nan
    if (walk / "walk_forward_summary.json").exists():
        walk_sr = read_json(walk / "walk_forward_summary.json")["annualized_sharpe"]
    text = f"""\\begin{{threeparttable}}
\\begin{{tabular}}{{lccc}}
\\toprule
Model & Validation SR & Test SR & Walk-forward SR \\\\
\\midrule
Neural factor model & {metrics['val']['sharpe']:.2f} & {metrics['test']['sharpe']:.2f} & {walk_sr:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}\\footnotesize
\\item The fixed test period is 2015--2024. Walk-forward results re-estimate the model before each test year.
\\end{{tablenotes}}
\\end{{threeparttable}}
"""
    write(paper / "tables" / "baseline_performance.tex", text)


def k_grid_table(robustness: Path, paper: Path):
    path = robustness / "robustness_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"{int(row.kmax)} & {row.test_sharpe_mean:.2f} & {row.test_sharpe_std:.2f} & "
            f"{row.entropy_rank_mean:.2f} & {100.0 * row.first_direction_share_mean:.1f}\\% \\\\"
        )
    text = """\\begin{threeparttable}
\\begin{tabular}{rrrrr}
\\toprule
$K_{\\max}$ & Mean Test SR & SD Test SR & Mean $K_{\\mathrm{eff}}$ & First Share \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item Ten random initializations per maximum factor capacity.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "k_grid.tex", text)


def factor_profile_table(baseline: Path, paper: Path):
    root = baseline / "interpretation"
    importance = pd.read_csv(root / "characteristic_importance.csv")
    profile = pd.read_csv(root / "dominant_factor_profile.csv")
    importance = importance.loc[importance["factor"] == 1, ["characteristic", "share"]]
    merged = importance.merge(profile[["characteristic", "top_minus_bottom"]], on="characteristic", how="left")
    merged = merged.sort_values("share", ascending=False).head(12)
    rows = [
        f"{row.characteristic.replace('_', '\\_')} & {100.0 * row.share:.2f}\\% & {row.top_minus_bottom:.3f} \\\\"
        for row in merged.itertuples()
    ]
    text = """\\begin{threeparttable}
\\begin{tabular}{lrr}
\\toprule
Characteristic & Sensitivity Share & Top--Bottom Tilt \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item Sensitivity is average absolute gradient share. The tilt is the mean characteristic-rank difference between the top and bottom factor-score deciles.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "factor_profile.tex", text)


def spanning_tables(baseline: Path, paper: Path):
    root = baseline / "spanning"
    spanning = pd.read_csv(root / "spanning_alpha.csv")
    hedged = pd.read_csv(root / "hedged_factor_sharpe.csv")
    marginal = pd.read_csv(root / "marginal_sharpe.csv")
    labels = {"FF5+MOM": "FF5 + MOM", "q5": "$q^5$", "QMJ": "QMJ", "Mispricing-JKP": "Mispricing", "JKP-core": "JKP core"}
    merged = spanning.merge(hedged, on="model", how="left")
    merged = merged.loc[merged["model"].isin(labels)]
    rows = []
    for row in merged.itertuples():
        rows.append(f"{labels[row.model]} & {row.alpha_annualized:.3f} & {row.alpha_t_nw:.2f} & {row.r2:.3f} & {row.hedged_sharpe:.2f} \\\\ ")
    text = """\\begin{threeparttable}
\\begin{tabular}{lrrrr}
\\toprule
Benchmark & Ann. Alpha & NW $t$ & $R^2$ & Hedged SR \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item The learned factor is scaled to 10\\% annual volatility using pre-test data. Newey--West standard errors use six lags.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "spanning.tex", text)
    mrows = []
    for row in marginal.loc[marginal["model"].isin(labels)].itertuples():
        mrows.append(
            f"{labels[row.model]} & {row.benchmark_sharpe:.2f} & {row.augmented_sharpe:.2f} & {row.marginal_sharpe:.2f} \\\\"
        )
    mtext = """\\begin{threeparttable}
\\begin{tabular}{lrrr}
\\toprule
Benchmark & Benchmark SR & + Learned Factor SR & Marginal SR \\\\
\\midrule
""" + "\n".join(mrows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item Tangency weights are estimated before 2015 and evaluated in the 2015--2024 holdout sample.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "marginal_sharpe.tex", mtext)


def costs_table(baseline: Path, paper: Path):
    root = baseline / "investability"
    summary = pd.read_csv(root / "summary.csv")
    monthly = pd.read_csv(root / "monthly_costs.csv")
    turnover = float(monthly["turnover_half_l1"].mean())
    labels = {
        "vol_target_return": "Gross, 10\\% vol target",
        "net_return_5bps": "5 bps",
        "net_return_10bps": "10 bps",
        "net_return_25bps": "25 bps",
    }
    rows = []
    for key, label in labels.items():
        block = summary.loc[summary["series"] == key]
        if block.empty:
            continue
        row = block.iloc[0]
        rows.append(
            f"{label} & {row.annualized_mean:.3f} & {row.annualized_vol:.3f} & {row.annualized_sharpe:.2f} & {turnover:.2f} \\\\"
        )
    text = """\\begin{threeparttable}
\\begin{tabular}{lrrrr}
\\toprule
Specification & Ann. Mean & Ann. Vol & Sharpe & Avg. Turnover \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item Volatility scaling uses lagged realized volatility and is not part of network training.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "costs.tex", text)


def ablation_table(baseline: Path, ablations: Path, paper: Path):
    base = read_json(baseline / "metrics.json")
    rows = [f"Full neural factor model & {base['val']['sharpe']:.2f} & {base['test']['sharpe']:.2f} \\\\"]
    path = ablations / "ablation_summary.csv"
    if path.exists():
        frame = pd.read_csv(path)
        labels = {
            "no_state": "No aggregate state $M_t$",
            "no_context": "No learned cross-sectional pooling",
            "static": "Static factor allocation",
        }
        for key in ("no_state", "no_context", "static"):
            block = frame.loc[frame["model"] == key]
            if len(block):
                row = block.iloc[0]
                rows.append(f"{labels[key]} & {row.val_sharpe:.2f} & {row.test_sharpe:.2f} \\\\")
    text = """\\begin{threeparttable}
\\begin{tabular}{lrr}
\\toprule
Model & Validation SR & Test SR \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item All specifications use the same characteristic information and maximum factor capacity.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "ablations.tex", text)


def stability_table(baseline: Path, paper: Path):
    root = baseline / "stability"
    dims_path = root / "economic_dimension_by_period.csv"
    comp_path = root / "interaction_stability.csv"
    if not dims_path.exists():
        return
    dims = pd.read_csv(dims_path)
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    rows = []
    labels = {
        "train_early": "Early training", "train_late": "Late training",
        "validation": "Validation", "test": "Test",
    }
    for row in dims.itertuples():
        # Use comparison with test when available to summarize interaction stability.
        corr = jaccard = np.nan
        if len(comp) and row.period != "test":
            mask = ((comp.period_1 == row.period) & (comp.period_2 == "test")) | ((comp.period_2 == row.period) & (comp.period_1 == "test"))
            if mask.any():
                c = comp.loc[mask].iloc[0]
                corr, jaccard = c.spearman_interaction_rank, c.top10_jaccard
        rows.append(f"{labels.get(row.period, row.period)} & {row.entropy_rank:.2f} & {100*row.first_direction_share:.1f}\\% & {corr:.2f} & {jaccard:.2f} \\\\ ")
    text = """\\begin{threeparttable}
\\begin{tabular}{lrrrr}
\\toprule
Period & $K_{\\mathrm{eff}}$ & First Share & Interaction Rank Corr. & Top-10 Overlap \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}\\footnotesize\\item Interaction stability compares each pre-test period with the 2015--2024 test period using a fixed characteristic set.\\end{tablenotes}
\\end{threeparttable}
"""
    write(paper / "tables" / "stability.tex", text)


def make_figures(baseline: Path, walk: Path, paper: Path):
    figdir = paper / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    spectrum = pd.read_csv(baseline / "factors" / "economic_factor_spectrum.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(spectrum["factor"], spectrum["share"])
    ax.set_xlabel("Canonical factor direction")
    ax.set_ylabel("Economic allocation share")
    ax.set_ylim(0, max(1.0, float(spectrum["share"].max()) * 1.05))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "economic_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)

    interactions_path = baseline / "interpretation" / "dominant_factor_interactions.csv"
    if interactions_path.exists():
        inter = pd.read_csv(interactions_path).head(12).iloc[::-1]
        labels = [f"{a} × {b}" for a, b in zip(inter.characteristic_1, inter.characteristic_2)]
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.barh(labels, inter["normalized_interaction"])
        ax.set_xlabel("Normalized nonlinear interaction")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figdir / "interactions.pdf", bbox_inches="tight")
        plt.close(fig)

    walk_path = walk / "walk_forward_returns.parquet"
    if walk_path.exists():
        wf = pd.read_parquet(walk_path).sort_values("date")
        wealth = (1.0 + wf["portfolio_return"]).cumprod()
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        ax.plot(wf["date"], wealth)
        ax.set_ylabel("Growth of $1")
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figdir / "walk_forward.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    baseline = Path(args.baseline)
    robustness = Path(args.robustness)
    ablations = Path(args.ablations)
    walk = Path(args.walk_forward)
    paper = Path(args.paper_dir)
    build_macros(baseline, robustness, walk, paper)
    baseline_table(baseline, walk, paper)
    k_grid_table(robustness, paper)
    factor_profile_table(baseline, paper)
    spanning_tables(baseline, paper)
    costs_table(baseline, paper)
    ablation_table(baseline, ablations, paper)
    stability_table(baseline, paper)
    make_figures(baseline, walk, paper)
    print("Paper tables and figures updated in", paper.resolve())


if __name__ == "__main__":
    main()
