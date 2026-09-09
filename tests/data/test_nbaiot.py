"""Phase 2 Part D: N-BaIoT preprocessing tests.

Mirrors test_cicids2017.py's structure. The key differences under
test here: client structure is fixed (device x shard), not simulated
via Dirichlet, and "chronological" means row order within a file, not
a real timestamp -- so the leakage check is per (device, label,
shard) rather than per source host, and there's an extra shard-order
check that has no CICIDS2017 equivalent.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fedpda_ids.data.common import assert_no_chronological_leakage

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_METADATA_PATH = REPO_ROOT / "data" / "processed" / "nbaiot" / "metadata.json"

ZERO_DAY_LABEL = "Mirai-Udpplain"


# ---------------------------------------------------------------------
# Schema / cleanliness
# ---------------------------------------------------------------------


def test_no_nan_or_inf_in_features(processed_nbaiot):
    df, metadata = processed_nbaiot
    features = df[metadata["feature_columns"]]
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy()).all()


def test_huge_magnitude_column_still_gets_log1p(processed_nbaiot):
    """Regression check for the float32 skew-overflow bug: a column with
    values around 1e15 (like the real HH_jit_*_variance features) must
    still be selected for log1p, not silently skipped because skew()
    overflowed to NaN."""
    _, metadata = processed_nbaiot
    assert "HH_jit_L5_variance" in metadata["log1p_columns"]


def test_client_id_columns_are_integer_dtype(processed_nbaiot):
    df, _ = processed_nbaiot
    assert df["client_id_45"].dtype.kind == "i"
    assert df["client_id_9"].dtype.kind == "i"


# ---------------------------------------------------------------------
# Zero-day holdout
# ---------------------------------------------------------------------


def test_zero_day_rows_excluded_from_every_client(processed_nbaiot):
    df, _ = processed_nbaiot
    holdout = df[df["is_zero_day_holdout"]]
    assert len(holdout) > 0
    assert (holdout["client_id_45"] == -1).all()
    assert (holdout["client_id_9"] == -1).all()
    assert (holdout["Label"] == ZERO_DAY_LABEL).all()


def test_non_holdout_rows_have_valid_client_id(processed_nbaiot):
    df, metadata = processed_nbaiot
    non_holdout = df[~df["is_zero_day_holdout"]]
    assert non_holdout["client_id_45"].between(0, metadata["num_clients_45"] - 1).all()
    assert non_holdout["client_id_9"].between(0, metadata["num_clients_9"] - 1).all()


def test_device_without_mirai_has_no_mirai_labels(processed_nbaiot):
    """Device_B mirrors the real Ennio_Doorbell / Samsung_SNH_1011_N_Webcam,
    which have no Mirai data at all -- this must not silently produce
    fabricated Mirai rows or crash the pipeline."""
    df, _ = processed_nbaiot
    device_b_labels = set(df.loc[df["device"] == "Device_B", "Label"].unique())
    assert not any(label.startswith("Mirai") for label in device_b_labels)


# ---------------------------------------------------------------------
# Leakage and shard chronology
# ---------------------------------------------------------------------


def test_no_chronological_leakage_within_any_shard(processed_nbaiot):
    df, _ = processed_nbaiot
    non_holdout = df[~df["is_zero_day_holdout"]]
    assert_no_chronological_leakage(
        non_holdout, group_col=["device", "Label", "shard_id"], time_col="row_order"
    )


def test_shards_are_chronologically_ordered(processed_nbaiot):
    """Shard 0 of a (device, label) must always be earlier in time than
    shard 1, etc. -- this is what makes a shard safe to build sliding
    windows from later."""
    df, _ = processed_nbaiot
    non_holdout = df[~df["is_zero_day_holdout"]]

    for _, group in non_holdout.groupby(["device", "Label"]):
        for shard_id in sorted(group["shard_id"].unique())[:-1]:
            this_shard = group.loc[group["shard_id"] == shard_id, "row_order"]
            next_shard = group.loc[group["shard_id"] == shard_id + 1, "row_order"]
            if len(this_shard) and len(next_shard):
                assert this_shard.max() <= next_shard.min()


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_same_seed_gives_identical_shard_and_split_assignment(synthetic_nbaiot_raw_dir, tmp_path):
    from fedpda_ids.data.nbaiot import preprocess_nbaiot

    kwargs = dict(
        raw_dir=synthetic_nbaiot_raw_dir,
        devices=["Device_A", "Device_B", "Device_C"],
        zero_day_holdout_labels=["Mirai-Udpplain"],
        train_fraction=0.70,
        val_fraction=0.15,
        log1p_skew_threshold=1.0,
        shards_per_device=5,
        seed=7,
    )
    meta_a = preprocess_nbaiot(processed_dir=tmp_path / "run_a", **kwargs)
    meta_b = preprocess_nbaiot(processed_dir=tmp_path / "run_b", **kwargs)

    import pandas as pd

    df_a = pd.read_parquet(tmp_path / "run_a" / "full_processed.parquet")
    df_b = pd.read_parquet(tmp_path / "run_b" / "full_processed.parquet")

    assert df_a["client_id_45"].tolist() == df_b["client_id_45"].tolist()
    assert df_a["temporal_split"].tolist() == df_b["temporal_split"].tolist()
    assert meta_a["label_counts"] == meta_b["label_counts"]


# ---------------------------------------------------------------------
# Real-data sanity check (skipped if you haven't run the real pipeline yet)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_METADATA_PATH.exists(),
    reason="real N-BaIoT preprocessing hasn't been run yet (data/processed/nbaiot/metadata.json missing)",
)
def test_real_metadata_matches_expectations():
    metadata = json.loads(REAL_METADATA_PATH.read_text())

    assert metadata["dropped_row_count"] == 0
    assert metadata["num_clients_45"] == 45
    assert metadata["num_clients_9"] == 9
    assert metadata["zero_day_holdout_row_count"] > 0
    assert metadata["num_features"] == 115
    assert "HH_jit_L5_variance" in metadata["log1p_columns"]
    assert len(metadata["devices"]) == 9
    assert set(metadata["label_counts"].keys()) == {
        "BENIGN",
        "BASHLITE-Combo", "BASHLITE-Junk", "BASHLITE-Scan", "BASHLITE-Tcp", "BASHLITE-Udp",
        "Mirai-Ack", "Mirai-Scan", "Mirai-Syn", "Mirai-Udp", "Mirai-Udpplain",
    }
