"""N-BaIoT preprocessing pipeline (Phase 2, Part C).

Unlike CICIDS2017, N-BaIoT's clients ARE real physical devices, fixed
by the frozen spec: 9 devices x 5 chronological shards = 45 clients,
plus a 9-client (no-shard) ablation. There is no Dirichlet partitioning
here -- client structure is given, not simulated.

Each device folder contains "benign_traffic.csv" plus "gafgyt_attacks/"
(BASHLITE) and "mirai_attacks/" CSVs -- all with the same 115 numeric
Kitsune-style statistical features, no identifier columns at all. Row
order within a file IS chronological order (features are computed via
streaming decay counters), so "preserve temporal ordering" here just
means: never shuffle rows, and never let a shard split a file's rows
out of order.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from fedpda_ids.data.common import (
    assign_chronological_split,
    drop_constant_columns,
    select_log1p_columns,
)

logger = logging.getLogger("fedpda_ids")

BENIGN_FILENAME = "benign_traffic.csv"
GAFGYT_SUBDIR = "gafgyt_attacks"
MIRAI_SUBDIR = "mirai_attacks"

BOOKKEEPING_COLUMNS = ["device", "label", "row_order"]


def _load_device(device_dir: Path, device_name: str) -> pd.DataFrame:
    """Load every label file for one device, tagging each row with its
    device, canonical label, and original within-file row position
    (row_order) -- the chronological-order surrogate, since these files
    have no timestamp column."""
    frames = []

    # dtype=float32 halves memory versus pandas' float64 default -- with
    # ~7M rows x 115 features this is the difference between the dataset
    # fitting comfortably in RAM and not, on a 16GB machine. PyTorch uses
    # float32 by default anyway, so nothing downstream loses precision
    # that would have mattered.
    benign_path = device_dir / BENIGN_FILENAME
    if benign_path.exists():
        df = pd.read_csv(benign_path, encoding="latin1", low_memory=False, dtype=np.float32)
        df["device"] = device_name
        df["label"] = "BENIGN"
        df["row_order"] = np.arange(len(df), dtype=np.int32)
        frames.append(df)

    for subdir, prefix in [(GAFGYT_SUBDIR, "BASHLITE"), (MIRAI_SUBDIR, "Mirai")]:
        attack_dir = device_dir / subdir
        if not attack_dir.exists():
            continue
        for csv_path in sorted(attack_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="latin1", low_memory=False, dtype=np.float32)
            df["device"] = device_name
            df["label"] = f"{prefix}-{csv_path.stem.capitalize()}"
            df["row_order"] = np.arange(len(df), dtype=np.int32)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No data files found under {device_dir}")

    return pd.concat(frames, ignore_index=True)


def load_raw_nbaiot(raw_dir: Path, devices: list[str]) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    missing = [d for d in devices if not (raw_dir / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing N-BaIoT device folders in {raw_dir}:\n  "
            + "\n  ".join(missing)
        )

    frames = []
    for device_name in devices:
        logger.info("Loading device %s", device_name)
        frames.append(_load_device(raw_dir / device_name, device_name))

    return pd.concat(frames, ignore_index=True)


def _assign_shards(df: pd.DataFrame, shards_per_device: int) -> pd.Series:
    """Split each (device, label) group into `shards_per_device`
    contiguous, chronologically-ordered chunks. Shard 0 is always
    earlier in time than shard 1, etc. -- this is what makes a shard
    safe to build sliding-window sequences from later."""
    # NOTE: pd.Series(index=..., dtype="int64") with no data silently
    # becomes float64 (pandas fills it with NaN first, which int64 can't
    # hold) -- every row gets overwritten below so the *values* end up
    # right regardless, but we cast back to int32 explicitly at the end
    # so shard_id/client_id columns don't end up as float in the output.
    shard = pd.Series(index=df.index, dtype="int64")

    for _, group_idx in df.groupby(["device", "label"]).groups.items():
        ordered = df.loc[group_idx, "row_order"].sort_values().index
        chunks = np.array_split(ordered, shards_per_device)
        for shard_id, chunk in enumerate(chunks):
            shard.loc[chunk] = shard_id

    return shard.astype(np.int32)


def preprocess_nbaiot(
    raw_dir: str | Path,
    processed_dir: str | Path,
    devices: list[str],
    zero_day_holdout_labels: list[str],
    train_fraction: float,
    val_fraction: float,
    log1p_skew_threshold: float,
    shards_per_device: int,
    seed: int,
) -> dict:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_nbaiot(raw_dir, devices)
    raw_row_count = len(df)

    feature_cols = [c for c in df.columns if c not in BOOKKEEPING_COLUMNS]

    # N-BaIoT's CSVs are pre-engineered numeric features, not raw flow
    # ratios like CICIDS2017 -- they loaded successfully with
    # dtype=float32 above, which itself proves there's no non-numeric
    # junk to coerce. The only thing left to check for is Inf/NaN, done
    # one column at a time (a few MB each) rather than via
    # coerce_numeric_and_flag_bad_rows' df.replace(), which internally
    # consolidates the whole ~3GB block into one contiguous array --
    # exactly the kind of extra full-size copy this dataset can't afford.
    clean_mask = np.ones(len(df), dtype=bool)
    for col in feature_cols:
        clean_mask &= np.isfinite(df[col].to_numpy())
    dropped_row_count = int((~clean_mask).sum())
    logger.info("Found %d non-finite (Inf/NaN) rows across %d total", dropped_row_count, len(df))

    if dropped_row_count:
        df = df.loc[clean_mask].reset_index(drop=True)
    del clean_mask
    gc.collect()

    df["is_zero_day_holdout"] = df["label"].isin(zero_day_holdout_labels)

    device_index = {name: i for i, name in enumerate(devices)}
    df["device_index"] = df["device"].map(device_index).astype(np.int32)

    df["shard_id"] = _assign_shards(df, shards_per_device)

    df["client_id_45"] = df["device_index"] * shards_per_device + df["shard_id"]
    df["client_id_9"] = df["device_index"]
    df.loc[df["is_zero_day_holdout"], ["client_id_45", "client_id_9"]] = -1

    # Chronological 70/15/15 split, per (device, label, shard) -- the
    # finest grouping -- so every client's test set stays representative
    # of every label it holds, instead of the split boundary landing
    # entirely inside whichever attack type happens to be chronologically
    # last (see Part C writeup for why a device-wide split would do that).
    #
    # Computed on a small 4-column slice (not the 115-feature block) and
    # merged back with .loc, rather than splitting df into two full
    # copies and re-concatenating them -- that pattern alone would have
    # tripled peak memory on this dataset's feature block.
    df["temporal_split"] = "zero_day_holdout"
    non_holdout_mask = ~df["is_zero_day_holdout"]
    split_result = assign_chronological_split(
        df.loc[non_holdout_mask, ["device", "label", "shard_id", "row_order"]],
        group_col=["device", "label", "shard_id"],
        time_col="row_order",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    df.loc[non_holdout_mask, "temporal_split"] = split_result
    del non_holdout_mask, split_result
    gc.collect()

    train_mask = df["temporal_split"] == "train"
    train_features = df.loc[train_mask, feature_cols]

    kept_cols = drop_constant_columns(train_features)
    dropped_constant_cols = sorted(set(feature_cols) - set(kept_cols))
    feature_cols = kept_cols
    del train_features
    train_features = df.loc[train_mask, feature_cols]

    log1p_cols = select_log1p_columns(train_features, log1p_skew_threshold)
    for col in log1p_cols:
        df[col] = np.log1p(df[col]).astype(np.float32)
    del train_features
    gc.collect()

    # Fit the scaler in row chunks via partial_fit rather than materializing
    # the whole ~2GB train-feature block as one array: min/max accumulate
    # correctly regardless of chunk boundaries, so this is exactly
    # equivalent to a single .fit() call, just with a much lower peak.
    scaler = MinMaxScaler()
    train_index = df.index[train_mask]
    for chunk_idx in np.array_split(train_index, 8):
        scaler.partial_fit(df.loc[chunk_idx, feature_cols].values)
    del train_mask, train_index
    gc.collect()

    # Apply the fitted scaler one feature column at a time instead of via
    # scaler.transform(df[feature_cols].values): that call would build a
    # full extra copy of the (already ~3GB) feature block, and sklearn can
    # upcast to float64 internally on top of that -- together enough to
    # exceed the ~7GB available on this machine. MinMaxScaler's transform
    # is exactly `X * scale_ + min_` per feature, so this is mathematically
    # identical, just column-at-a-time (a few MB at a time instead of GB).
    for i, col in enumerate(feature_cols):
        df[col] = (
            df[col].to_numpy(dtype=np.float32) * np.float32(scaler.scale_[i])
            + np.float32(scaler.min_[i])
        )

    output_path = processed_dir / "full_processed.parquet"
    df.to_parquet(output_path, index=False)

    scaler_path = processed_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    metadata = {
        "dataset": "nbaiot",
        "seed": seed,
        "raw_row_count": raw_row_count,
        "dropped_row_count": dropped_row_count,
        "kept_row_count": len(df),
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "dropped_constant_columns": dropped_constant_cols,
        "log1p_columns": log1p_cols,
        "devices": devices,
        "shards_per_device": shards_per_device,
        "num_clients_45": len(devices) * shards_per_device,
        "num_clients_9": len(devices),
        "zero_day_holdout_labels": zero_day_holdout_labels,
        "zero_day_holdout_row_count": int(df["is_zero_day_holdout"].sum()),
        "label_counts": df["label"].value_counts().to_dict(),
        "device_counts": df["device"].value_counts().to_dict(),
        "split_counts": df["temporal_split"].value_counts().to_dict(),
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "output_path": str(output_path),
        "scaler_path": str(scaler_path),
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata
