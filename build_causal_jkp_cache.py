from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

META_COLUMNS = [
    "id", "eom", "excntry", "gvkey", "permno", "size_grp", "me",
    "ret_exc_lead1m",
]
START_DATE = pd.Timestamp("1963-01-01")
END_DATE = pd.Timestamp("2024-12-31")
DEFAULT_TRAIN_END = pd.Timestamp("2004-11-30")
MAX_CHARACTERISTIC_MISSING = 1.0 / 3.0
MAX_ROW_MISSING = 0.30


def parse_args():
    parser = argparse.ArgumentParser(description="Build a training-only JKP feature universe")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end", default=str(DEFAULT_TRAIN_END.date()))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def coerce(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if not is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def sample_mask(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["eom"])
    return (
        dates.between(START_DATE, END_DATE)
        & frame["size_grp"].notna()
        & frame["size_grp"].ne("nano")
    )


def rank_standardize(frame: pd.DataFrame, columns: list[str]) -> None:
    grouped = frame.groupby("eom", sort=False)[columns]
    ranks = grouped.rank(method="average", na_option="keep")
    counts = grouped.transform("count")
    scaled = (ranks - 1.0).div(counts - 1.0) - 0.5
    frame.loc[:, columns] = scaled.mask(counts <= 1).fillna(0.0)


def raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("JKP_USA_*.parquet"))
    if not files:
        raise FileNotFoundError(raw_dir)
    return files


def select_characteristics(files: list[Path], train_end: pd.Timestamp):
    first = pd.read_parquet(files[0])
    characteristics = [c for c in first.columns if c not in META_COLUMNS]
    non_missing = pd.Series(0, index=characteristics, dtype="int64")
    observations = 0
    for path in files:
        year = int(path.stem.rsplit("_", 1)[1])
        if year < START_DATE.year:
            continue
        if year > train_end.year:
            break
        frame = pd.read_parquet(path, columns=["eom", "size_grp", *characteristics])
        frame["eom"] = pd.to_datetime(frame["eom"])
        coerce(frame, characteristics)
        mask = sample_mask(frame) & frame["eom"].le(train_end)
        values = frame.loc[mask, characteristics]
        observations += len(values)
        non_missing = non_missing.add(values.notna().sum(), fill_value=0)
        print(f"coverage {year}: {observations:,} observations", flush=True)
    missing_rate = 1.0 - non_missing / observations
    kept = missing_rate[missing_rate <= MAX_CHARACTERISTIC_MISSING].index.tolist()
    if not kept:
        raise RuntimeError("No characteristics pass the training-only coverage screen")
    return kept, missing_rate, observations


def build_cache(raw_dir: Path, output_dir: Path, train_end: pd.Timestamp, force: bool):
    files = raw_files(raw_dir)
    manifest_path = output_dir / "cleaning_manifest.json"
    if manifest_path.exists() and not force:
        print(f"Cache already exists: {output_dir}")
        return
    kept, missing_rate, observations = select_characteristics(files, train_end)
    print(f"Training-only characteristic universe: {len(kept)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    rows_after_cleaning = 0
    max_missing_count = len(kept) * MAX_ROW_MISSING
    load_columns = [*META_COLUMNS, *kept]
    for path in files:
        year = int(path.stem.rsplit("_", 1)[1])
        if year < START_DATE.year or year > END_DATE.year:
            continue
        frame = pd.read_parquet(path, columns=load_columns)
        frame["eom"] = pd.to_datetime(frame["eom"])
        coerce(frame, kept)
        frame = frame.loc[sample_mask(frame)].copy()
        row_missing = frame[kept].isna().sum(axis=1)
        frame = frame.loc[row_missing <= max_missing_count].copy()
        rank_standardize(frame, kept)
        frame = frame.sort_values(["id", "eom"], kind="stable").reset_index(drop=True)
        name = f"JKP_USA_clean_{year}.parquet"
        frame.to_parquet(output_dir / name, index=False, compression="zstd")
        output_files.append(name)
        rows_after_cleaning += len(frame)
        print(f"write {year}: {len(frame):,}", flush=True)
    manifest = {
        "cleaning_version": "training_only_v1",
        "selection_sample_start": str(START_DATE.date()),
        "selection_sample_end": str(train_end.date()),
        "selection_observations": int(observations),
        "max_characteristic_missing_share": MAX_CHARACTERISTIC_MISSING,
        "max_row_missing_share": MAX_ROW_MISSING,
        "rows_after_cleaning": int(rows_after_cleaning),
        "kept_characteristics": kept,
        "missing_rates_training": {key: float(value) for key, value in missing_rate.items()},
        "output_files": output_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote causal cache: {output_dir}", flush=True)


def main():
    args = parse_args()
    build_cache(
        Path(args.raw_dir),
        Path(args.output_dir),
        pd.Timestamp(args.train_end),
        args.force,
    )


if __name__ == "__main__":
    main()
