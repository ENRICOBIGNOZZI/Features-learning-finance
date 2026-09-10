from __future__ import annotations

import argparse
from pathlib import Path

from neural_factors.investability import analyze_investability


def parse_args():
    parser = argparse.ArgumentParser(description="Turnover, costs, and ex-post volatility targeting")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[5.0, 10.0, 25.0])
    parser.add_argument("--borrow-bps-annual", type=float, nargs="+", default=[100.0, 300.0])
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    output = result_dir / "investability"
    output.mkdir(parents=True, exist_ok=True)
    result = analyze_investability(
        result_dir=result_dir,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        cost_bps=tuple(args.cost_bps),
        target_annual_vol=args.target_vol,
        borrow_bps_annual=tuple(args.borrow_bps_annual),
    )
    result["monthly"].to_csv(output / "monthly_costs.csv", index=False)
    result["summary"].to_csv(output / "summary.csv", index=False)
    result["weights"].to_parquet(output / "scaled_weights.parquet", index=False)
    print(result["summary"].to_string(index=False))
    print("\nAverage half-L1 turnover:", result["monthly"]["turnover_half_l1"].mean())
    print("Average gross exposure:", result["monthly"]["gross_exposure"].mean())
    print("Average short exposure:", result["monthly"]["short_exposure"].mean())


if __name__ == "__main__":
    main()
