from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neural_factors.data import JKPUSData
from neural_factors.interpret import (
    allocator_state_links,
    factor_profile,
    gradient_importance,
    load_model,
    panelwise_permutation_interactions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Interpret active canonical neural factors")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--start", default="2014-12-31")
    parser.add_argument("--end", default="2024-11-30")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--max-factors", type=int, default=3)
    parser.add_argument("--cumulative-share", type=float, default=0.95)
    return parser.parse_args()


def active_factor_count(result_dir: Path, threshold: float, maximum: int) -> int:
    spectrum = pd.read_csv(result_dir / "factors" / "economic_factor_spectrum.csv")
    cumulative = spectrum["share"].cumsum().to_numpy(dtype=float)
    required = int(np.searchsorted(cumulative, threshold) + 1)
    return max(1, min(required, maximum, len(spectrum)))


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = result_dir / "interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    model, scaler, transform = load_model(result_dir, data)
    n_active = active_factor_count(
        result_dir, args.cumulative_share, args.max_factors
    )
    importance = gradient_importance(
        model, data, scaler, transform, args.start, args.end,
        n_factors=n_active, max_months=36,
    )
    importance.to_csv(output_dir / "characteristic_importance.csv", index=False)

    profiles = []
    interactions = []
    for factor_index in range(n_active):
        factor_importance = importance.loc[importance["factor"] == factor_index + 1]
        top_characteristics = factor_importance.head(args.top)["characteristic"].tolist()
        profile = factor_profile(
            model, data, scaler, transform, args.start, args.end,
            factor=factor_index,
        )
        profiles.append(profile)
        interaction = panelwise_permutation_interactions(
            model, data, scaler, transform, args.start, args.end,
            top_characteristics, factor=factor_index, max_months=18,
            seed=42 + factor_index,
        )
        interactions.append(interaction)

    profiles_frame = pd.concat(profiles, ignore_index=True)
    interactions_frame = pd.concat(interactions, ignore_index=True)
    profiles_frame.to_csv(output_dir / "factor_profiles.csv", index=False)
    interactions_frame.to_csv(output_dir / "factor_interactions.csv", index=False)
    state_links = allocator_state_links(data, result_dir, split="test")
    state_links.to_csv(output_dir / "allocator_state_links.csv", index=False)

    print(f"Active canonical factors covering {args.cumulative_share:.0%}: {n_active}")
    for factor_index in range(n_active):
        factor_no = factor_index + 1
        print(f"\nFactor {factor_no}: top characteristic sensitivities")
        print(
            importance.loc[importance["factor"] == factor_no]
            .head(args.top).to_string(index=False)
        )
        print(f"\nFactor {factor_no}: top nonlinear interactions")
        print(
            interactions_frame.loc[interactions_frame["factor"] == factor_no]
            .head(args.top).to_string(index=False)
        )
        print(f"\nFactor {factor_no}: top characteristic tilts")
        print(
            profiles_frame.loc[profiles_frame["factor"] == factor_no]
            .head(args.top).to_string(index=False)
        )


if __name__ == "__main__":
    main()
