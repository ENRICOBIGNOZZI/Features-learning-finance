from __future__ import annotations

import argparse
from pathlib import Path

from neural_factors.data import JKPUSData
from neural_factors.interpret import (
    allocator_state_links,
    collect_stock_sample,
    dominant_factor_profile,
    gradient_importance,
    load_model,
    permutation_interactions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Interpret canonical neural factors")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--sample-rows", type=int, default=6000)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = result_dir / "interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = JKPUSData(args.data_dir)
    model, scaler, transform = load_model(result_dir, data)
    importance = gradient_importance(
        model, data, scaler, transform, args.start, args.end, n_factors=min(3, transform.shape[1])
    )
    importance.to_csv(output_dir / "characteristic_importance.csv", index=False)

    dominant = importance.loc[importance["factor"] == 1]
    top_characteristics = dominant.head(args.top)["characteristic"].tolist()
    sample = collect_stock_sample(
        data, scaler, args.start, args.end, max_rows=args.sample_rows
    )
    interactions = permutation_interactions(
        model,
        sample,
        transform,
        data.characteristics,
        top_characteristics,
        factor=0,
    )
    interactions.to_csv(output_dir / "dominant_factor_interactions.csv", index=False)

    profile = dominant_factor_profile(
        model, data, scaler, transform, args.start, args.end
    )
    profile.to_csv(output_dir / "dominant_factor_profile.csv", index=False)
    state_links = allocator_state_links(data, result_dir, split="test")
    state_links.to_csv(output_dir / "allocator_state_links.csv", index=False)

    print("Top dominant-factor characteristics")
    print(dominant.head(args.top).to_string(index=False))
    print("\nTop nonlinear interactions")
    print(interactions.head(args.top).to_string(index=False))
    print("\nTop long-minus-short characteristic tilts")
    print(profile.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
