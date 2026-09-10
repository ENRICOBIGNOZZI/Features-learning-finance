from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_factors.analysis import economic_dimension, save_canonical_analysis
from neural_factors.data import JKPUSData
from neural_factors.model import NeuralFactorModel
from neural_factors.policy_spectrum import save_policy_spectrum
from neural_factors.train import SplitConfig, evaluate, choose_device


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize diagnostics for a saved NN checkpoint")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_checkpoint(run_dir: Path, data: JKPUSData, device: torch.device):
    config = json.loads((run_dir / "config.json").read_text())
    train = config["train"]
    model = NeuralFactorModel(
        input_dim=data.input_dim, state_dim=len(data.state_columns),
        hidden=train["hidden"], factors=train["factors"],
        context_heads=train["context_heads"],
        use_context=config.get("use_context", True),
        use_state=config.get("use_state", True),
        static_allocator=config.get("static_allocator", False),
        linear_characteristics=config.get("linear_characteristics", False),
        conditional_scores=config.get("conditional_scores", False),
    ).to(device)
    state_dict = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    scaler = (
        np.asarray(config["state_mean"], dtype=np.float32),
        np.asarray(config["state_std"], dtype=np.float32),
    )
    split = SplitConfig(**config["split"])
    return model, scaler, split, config


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    data = JKPUSData(args.data_dir)
    device = choose_device(args.device)
    model, scaler, split, config = load_checkpoint(run_dir, data, device)
    ranges = {
        "train": (split.train_start, split.train_end),
        "val": (split.val_start, split.val_end),
        "test": (split.test_start, split.test_end),
    }
    frames = {}
    metrics = {}
    batch_months = int(config["train"].get("batch_months", 12))
    for name, (start, end) in ranges.items():
        split_metrics, frame = evaluate(
            model, data, start, end, scaler, device, batch_months=batch_months
        )
        metrics[name] = split_metrics
        frames[name] = frame
        frame.to_parquet(run_dir / f"monthly_{name}.parquet", index=False)

    basis = save_canonical_analysis(frames["train"], frames, run_dir / "factors")
    metrics["allocation_dimension"] = economic_dimension(basis.economic_eigenvalues)
    metrics["policy_dimension"] = save_policy_spectrum(
        model, data, split.train_start, split.train_end, scaler, device,
        run_dir / "policy", batch_months=batch_months,
    )
    (run_dir / "materialized_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
