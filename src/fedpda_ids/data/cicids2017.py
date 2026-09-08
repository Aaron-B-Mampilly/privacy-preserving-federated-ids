"""CICIDS2017 preprocessing pipeline (Phase 2, Part A).

Turns the 8 official weekday CSVs into one cleaned, labeled, scaled
table with the metadata later phases need (source host + timestamp for
Phase 3 sequence construction, temporal train/val/test split, federated
client assignment for 3 Dirichlet alpha values), WITHOUT building any
LSTM sequences yet -- that is Phase 3.

Identifier columns (Flow ID, Source Port, Destination IP) are dropped
entirely. Source IP and Timestamp are kept as METADATA columns -- not
model features -- because Phase 3 needs them to group and sort flows
into genuine per-host temporal sequences. "Not a model feature" and
"not present in the processed table" are different things; the frozen
spec asks for the former.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fedpda_ids.data.common import (
    assign_chronological_split,
    coerce_numeric_and_flag_bad_rows,
    dirichlet_partition_with_splits,
    drop_constant_columns,
    fit_minmax_scaler,
    select_log1p_columns,
)

logger = logging.getLogger("fedpda_ids")

# Columns dropped entirely: pure identifiers with no analytical or
# grouping value for this project (Destination IP is not used for
# grouping -- only Source IP is, per the frozen sequence-construction spec).
DROP_COLUMNS = ["Flow ID", "Source Port", "Destination IP"]

# Kept as metadata (never fed to the model as a feature).
METADATA_COLUMNS = ["Source IP", "Timestamp"]

LABEL_COLUMN = "Label"
PORT_COLUMN = "Destination Port"

# (substring match on a lowercased, whitespace-collapsed label) -> canonical name.
# Substring matching (not exact equality) because CICIDS2017's raw CSVs
# encode the en-dash in "Web Attack – X" inconsistently across files/tools,
# which corrupts exact string matches but not substring matches on the
# surrounding words.
_LABEL_RULES: list[tuple[list[str], str]] = [
    (["benign"], "BENIGN"),
    (["ftp-patator"], "FTP-Patator"),
    (["ftp", "patator"], "FTP-Patator"),
    (["ssh-patator"], "SSH-Patator"),
    (["ssh", "patator"], "SSH-Patator"),
    (["heartbleed"], "Heartbleed"),
    (["dos", "hulk"], "DoS Hulk"),
    (["dos", "goldeneye"], "DoS GoldenEye"),
    (["dos", "slowloris"], "DoS slowloris"),
    (["dos", "slowhttptest"], "DoS Slowhttptest"),
    (["web attack", "brute"], "Web Attack-Brute Force"),
    (["web attack", "xss"], "Web Attack-XSS"),
    (["web attack", "sql"], "Web Attack-SQL Injection"),
    (["infiltration"], "Infiltration"),
    (["bot"], "Bot"),
    (["portscan"], "PortScan"),
    (["ddos"], "DDoS"),
]


def _normalize_label(raw_label: str) -> str:
    text = str(raw_label).strip().lower()
    for keywords, canonical in _LABEL_RULES:
        if all(k in text for k in keywords):
            return canonical
    raise ValueError(f"Unrecognized CICIDS2017 label: {raw_label!r}")


def _bucket_destination_port(port: pd.Series) -> pd.Series:
    """well-known (0-1023) / registered (1024-49151) / dynamic (49152-65535)."""
    bucketed = pd.cut(
        port,
        bins=[-1, 1023, 49151, 65535],
        labels=[0, 1, 2],
    )
    return bucketed.astype(float).astype(int)


def load_raw_cicids2017(raw_dir: Path, file_to_day: dict[str, str]) -> pd.DataFrame:
    """Read and concatenate every configured weekday CSV, tagging each
    row with its weekday. Missing files are reported clearly rather
    than failing with a raw pandas FileNotFoundError."""
    raw_dir = Path(raw_dir)
    missing = [f for f in file_to_day if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CICIDS2017 raw files in "
            f"{raw_dir}:\n  " + "\n  ".join(missing) + "\n\n"
            "Download the official CIC-IDS-2017 CSVs (MachineLearningCVE "
            "release) from the UNB CIC dataset page and place them, "
            "unmodified, in this directory."
        )

    frames = []
    for filename, day in file_to_day.items():
        path = raw_dir / filename
        logger.info("Loading %s (%s)", filename, day)
        # latin1 decodes any byte sequence, avoiding UnicodeDecodeError on
        # the dataset's inconsistently-encoded en-dash in attack labels;
        # label text is recovered via substring matching in _normalize_label.
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["__source_file__"] = filename
        df["__day__"] = day
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # The raw CSVs contain a duplicated "Fwd Header Length" column; pandas
    # renames the second occurrence to "Fwd Header Length.1" on read. It's
    # an exact duplicate of the first, so drop it rather than double-count it.
    duplicate_cols = [c for c in combined.columns if c.endswith(".1")]
    if duplicate_cols:
        logger.info("Dropping duplicate columns: %s", duplicate_cols)
        combined = combined.drop(columns=duplicate_cols)

    # Some official CICIDS2017 releases (notably the Thursday-Morning
    # WebAttacks file) contain a large trailing block of fully blank rows
    # -- every column NaN, including Label -- which appears to be a
    # padding/export artifact rather than real flow data. Drop it here,
    # before label normalization, rather than let it crash label parsing.
    real_cols = [c for c in combined.columns if c not in ("__source_file__", "__day__")]
    blank_mask = combined[real_cols].isna().all(axis=1)
    if blank_mask.any():
        logger.info(
            "Dropping %d fully blank rows (known raw-file artifact, not real flow data)",
            int(blank_mask.sum()),
        )
        combined = combined.loc[~blank_mask].reset_index(drop=True)

    return combined


def preprocess_cicids2017(
    raw_dir: str | Path,
    processed_dir: str | Path,
    file_to_day: dict[str, str],
    zero_day_holdout_labels: list[str],
    train_fraction: float,
    val_fraction: float,
    log1p_skew_threshold: float,
    dirichlet_alpha_values: list[float],
    num_clients: int,
    seed: int,
) -> dict:
    """Run the full Part A pipeline. Returns the metadata dict that also
    gets written to processed_dir/metadata.json."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_cicids2017(raw_dir, file_to_day)
    raw_row_count = len(df)

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(_normalize_label)
    # The 8 raw files don't all use the same timestamp format (Monday has
    # seconds and zero-padding, e.g. "03/07/2017 08:55:58"; the rest don't,
    # e.g. "7/7/2017 8:59"). All of it is day-first (the capture ran
    # 03-07 July 2017, Mon-Fri). format="mixed" parses each value with its
    # own inferred format instead of locking the whole column to whichever
    # format the first row happens to have -- which is what silently turned
    # every non-Monday timestamp into NaT before this fix.
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce", dayfirst=True, format="mixed"
    )

    feature_cols = [
        c
        for c in df.columns
        if c not in METADATA_COLUMNS
        and c not in ("__source_file__", "__day__", LABEL_COLUMN)
    ]

    df, clean_mask = coerce_numeric_and_flag_bad_rows(df, feature_cols)
    valid_mask = clean_mask & df["Timestamp"].notna()
    dropped_row_count = int((~valid_mask).sum())
    df = df.loc[valid_mask].reset_index(drop=True)

    df[PORT_COLUMN] = _bucket_destination_port(df[PORT_COLUMN])

    df["is_zero_day_holdout"] = df[LABEL_COLUMN].isin(zero_day_holdout_labels)

    non_holdout = df.loc[~df["is_zero_day_holdout"]].copy()
    holdout = df.loc[df["is_zero_day_holdout"]].copy()

    non_holdout["temporal_split"] = assign_chronological_split(
        non_holdout,
        group_col="Source IP",
        time_col="Timestamp",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    holdout["temporal_split"] = "zero_day_holdout"

    df = pd.concat([non_holdout, holdout], ignore_index=True)

    train_mask = df["temporal_split"] == "train"
    train_features = df.loc[train_mask, feature_cols]

    kept_cols = drop_constant_columns(train_features)
    dropped_constant_cols = sorted(set(feature_cols) - set(kept_cols))
    feature_cols = kept_cols
    train_features = df.loc[train_mask, feature_cols]

    log1p_cols = select_log1p_columns(train_features, log1p_skew_threshold)
    # Destination Port is a 3-value categorical bucket (well-known/registered/
    # dynamic), not a continuous count -- log1p on it would just be a
    # monotonic relabeling with no distributional benefit, so it's excluded
    # even though its skew could technically clear the threshold.
    log1p_cols = [c for c in log1p_cols if c != PORT_COLUMN]
    for col in log1p_cols:
        df[col] = np.log1p(df[col])
    train_features = df.loc[train_mask, feature_cols]

    scaler = fit_minmax_scaler(train_features)
    df[feature_cols] = scaler.transform(df[feature_cols].values)

    client_id_columns = []
    for alpha in dirichlet_alpha_values:
        col_name = f"client_id_alpha{alpha}"
        client_id_columns.append(col_name)
        labels_for_partition = df[LABEL_COLUMN].where(~df["is_zero_day_holdout"])
        df[col_name] = dirichlet_partition_with_splits(
            labels=labels_for_partition,
            split=df["temporal_split"],
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )

    output_path = processed_dir / "full_processed.parquet"
    df.to_parquet(output_path, index=False)

    scaler_path = processed_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    metadata = {
        "dataset": "cicids2017",
        "seed": seed,
        "raw_row_count": raw_row_count,
        "dropped_row_count": dropped_row_count,
        "kept_row_count": len(df),
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "metadata_columns": METADATA_COLUMNS,
        "dropped_constant_columns": dropped_constant_cols,
        "log1p_columns": log1p_cols,
        "zero_day_holdout_labels": zero_day_holdout_labels,
        "zero_day_holdout_row_count": int(df["is_zero_day_holdout"].sum()),
        "label_counts": df[LABEL_COLUMN].value_counts().to_dict(),
        "day_counts": df["__day__"].value_counts().to_dict(),
        "split_counts": df["temporal_split"].value_counts().to_dict(),
        "dirichlet_alpha_values": dirichlet_alpha_values,
        "client_id_columns": client_id_columns,
        "num_clients": num_clients,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "output_path": str(output_path),
        "scaler_path": str(scaler_path),
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata
