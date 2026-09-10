from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from neural_factors.analysis import economic_dimension, fit_canonical_basis
from neural_factors.data import JKPUSData
from neural_factors.interpret import (
    gradient_importance,
    load_model,
    panelwise_permutation_interactions,
)
from neural_factors.policy_spectrum import effective_rank, estimate_policy_spectrum


def parse_args():
    parser = argparse.ArgumentParser(description="Factor dimension and interaction stability")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--top-characteristics", type=int, default=8)
    return parser.parse_args()


def periods():
    # Formation-date windows aligned to economically clean payoff periods.
    return {
        "train_early": ("1963-01-31", "1984-11-30"),
        "train_late": ("1984-12-31", "2004-11-30"),
        "validation": ("2004-12-31", "2014-11-30"),
        "test": ("2014-12-31", "2024-11-30"),
    }


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = result_dir / "stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    model, scaler, transform = load_model(result_dir, data)

    # Select the comparison set using training data only; the test sample is never
    # used to decide which interactions will be inspected.
    selection_imp = gradient_importance(
        model, data, scaler, transform, "1984-12-31", "2004-11-30",
        n_factors=1, max_months=24,
    )
    top_chars = selection_imp.head(args.top_characteristics)["characteristic"].tolist()
    selection_imp.to_csv(output_dir / "training_characteristic_importance.csv", index=False)

    interaction_tables = {}
    for name, (start, end) in periods().items():
        interactions = panelwise_permutation_interactions(
            model, data, scaler, transform, start, end, top_chars,
            factor=0, max_months=12, seed=42,
        )
        interactions["period"] = name
        interaction_tables[name] = interactions
    pd.concat(interaction_tables.values(), ignore_index=True).to_csv(
        output_dir / "interaction_scores_by_period.csv", index=False
    )

    monthly = pd.concat(
        [pd.read_parquet(result_dir / f"monthly_{split}.parquet") for split in ("train", "val", "test")],
        ignore_index=True,
    ).sort_values("date")
    dim_rows = []
    for name, (start, end) in periods().items():
        block = monthly.loc[monthly["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        basis = fit_canonical_basis(block)
        dims = economic_dimension(basis.economic_eigenvalues)
        values = basis.economic_eigenvalues
        shares = values / max(values.sum(), 1e-12)
        dim_rows.append({
            "period": name,
            "months": len(block),
            **dims,
            "first_direction_share": float(shares[0]),
            "top2_share": float(shares[:2].sum()),
        })
    dimensions = pd.DataFrame(dim_rows)
    dimensions.to_csv(output_dir / "allocation_dimension_by_period.csv", index=False)

    pair_rows = []
    for first, second in combinations(interaction_tables, 2):
        a = interaction_tables[first].copy()
        b = interaction_tables[second].copy()
        key = ["characteristic_1", "characteristic_2"]
        merged = a[key + ["normalized_interaction"]].merge(
            b[key + ["normalized_interaction"]], on=key, suffixes=("_a", "_b")
        )
        spearman = merged["normalized_interaction_a"].corr(
            merged["normalized_interaction_b"], method="spearman"
        )
        top_n = min(10, len(merged))
        top_a = set(map(tuple, a.head(top_n)[key].to_numpy()))
        top_b = set(map(tuple, b.head(top_n)[key].to_numpy()))
        union = top_a | top_b
        pair_rows.append({
            "period_1": first,
            "period_2": second,
            "spearman_interaction_rank": float(spearman),
            "top10_jaccard": float(len(top_a & top_b) / max(len(union), 1)),
        })
    comparison = pd.DataFrame(pair_rows)
    comparison.to_csv(output_dir / "interaction_stability.csv", index=False)

    print("Fixed characteristic set:", top_chars)
    print("\nAllocation effective rank by period")
    print(dimensions.to_string(index=False))
    print("\nInteraction stability")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
