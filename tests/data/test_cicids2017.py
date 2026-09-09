"""Phase 2 Part B: CICIDS2017 preprocessing tests.

These validate the pipeline against synthetic data on every run
(fast, deterministic). A separate, slower test additionally checks
the metadata.json from your real preprocessing run, when present, so
we also confirm the real numbers make sense.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedpda_ids.data.common import assert_no_chronological_leakage

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_METADATA_PATH = REPO_ROOT / "data" / "processed" / "cicids2017" / "metadata.json"

EXPECTED_LABELS = {
    "BENIGN",
    "FTP-Patator",
    "SSH-Patator",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Heartbleed",
    "Web Attack-Brute Force",
    "Web Attack-XSS",
    "Web Attack-SQL Injection",
    "Infiltration",
    "Bot",
    "PortScan",
    "DDoS",
}

ZERO_DAY_LABELS = {"Infiltration", "Web Attack-Brute Force"}


# ---------------------------------------------------------------------
# Schema / cleanliness
# ---------------------------------------------------------------------


def test_no_nan_or_inf_in_features(processed_cicids):
    df, metadata = processed_cicids
    features = df[metadata["feature_columns"]]
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy()).all()


def test_labels_are_canonical(processed_cicids):
    df, _ = processed_cicids
    assert set(df["Label"].unique()) <= EXPECTED_LABELS


def test_metadata_and_model_features_do_not_overlap(processed_cicids):
    df, metadata = processed_cicids
    assert set(metadata["metadata_columns"]).isdisjoint(metadata["feature_columns"])
    for col in ("Flow ID", "Source Port", "Destination IP"):
        assert col not in df.columns


# ---------------------------------------------------------------------
# Zero-day holdout
# ---------------------------------------------------------------------


def test_zero_day_labels_are_exactly_the_configured_ones(processed_cicids):
    df, _ = processed_cicids
    holdout_labels = set(df.loc[df["is_zero_day_holdout"], "Label"].unique())
    assert holdout_labels <= ZERO_DAY_LABELS


def test_zero_day_rows_excluded_from_every_client(processed_cicids):
    df, metadata = processed_cicids
    holdout = df[df["is_zero_day_holdout"]]
    for col in metadata["client_id_columns"]:
        assert (holdout[col] == -1).all()


def test_non_holdout_rows_have_valid_client_id(processed_cicids):
    df, metadata = processed_cicids
    non_holdout = df[~df["is_zero_day_holdout"]]
    for col in metadata["client_id_columns"]:
        assert non_holdout[col].between(0, metadata["num_clients"] - 1).all()


# ---------------------------------------------------------------------
# Leakage: the check that matters most for Phase 3
# ---------------------------------------------------------------------


def test_no_chronological_leakage_within_any_host(processed_cicids):
    df, _ = processed_cicids
    non_holdout = df[~df["is_zero_day_holdout"]]
    assert_no_chronological_leakage(non_holdout, group_col="Source IP", time_col="Timestamp")


# ---------------------------------------------------------------------
# Dirichlet partitioning: must actually control non-IID severity
# ---------------------------------------------------------------------


def test_lower_alpha_produces_more_client_label_skew(processed_cicids):
    """This is the mechanism E2 (non-IID severity) depends on -- if this
    is backwards, the whole non-IID experiment is invalid."""
    df, metadata = processed_cicids
    non_holdout = df[~df["is_zero_day_holdout"]]

    def benign_fraction_std(alpha_col: str) -> float:
        fractions = non_holdout.groupby(alpha_col)["Label"].apply(
            lambda s: (s == "BENIGN").mean()
        )
        return fractions.std()

    alpha_to_col = dict(zip(metadata["dirichlet_alpha_values"], metadata["client_id_columns"]))
    std_alpha_5 = benign_fraction_std(alpha_to_col[5])
    std_alpha_0_1 = benign_fraction_std(alpha_to_col[0.1])

    assert std_alpha_0_1 > std_alpha_5


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_same_seed_gives_identical_client_assignment(synthetic_cicids_raw_dir, tmp_path):
    from fedpda_ids.data.cicids2017 import preprocess_cicids2017

    file_to_day = {
        "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
        "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
        "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday",
    }
    kwargs = dict(
        raw_dir=synthetic_cicids_raw_dir,
        file_to_day=file_to_day,
        zero_day_holdout_labels=["Infiltration", "Web Attack-Brute Force"],
        train_fraction=0.70,
        val_fraction=0.15,
        log1p_skew_threshold=1.0,
        dirichlet_alpha_values=[5, 0.5, 0.1],
        num_clients=40,
        seed=123,
    )
    meta_a = preprocess_cicids2017(processed_dir=tmp_path / "run_a", **kwargs)
    meta_b = preprocess_cicids2017(processed_dir=tmp_path / "run_b", **kwargs)

    df_a = pd.read_parquet(tmp_path / "run_a" / "full_processed.parquet")
    df_b = pd.read_parquet(tmp_path / "run_b" / "full_processed.parquet")

    for col in meta_a["client_id_columns"]:
        assert df_a[col].tolist() == df_b[col].tolist()


# ---------------------------------------------------------------------
# Real-data sanity check (skipped if you haven't run the real pipeline yet)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_METADATA_PATH.exists(),
    reason="real CICIDS2017 preprocessing hasn't been run yet (data/processed/cicids2017/metadata.json missing)",
)
def test_real_metadata_matches_expectations():
    metadata = json.loads(REAL_METADATA_PATH.read_text())

    # dropped rows should be a small fraction of raw rows, not a mass-drop
    drop_fraction = metadata["dropped_row_count"] / metadata["raw_row_count"]
    assert drop_fraction < 0.01, (
        f"{drop_fraction:.2%} of rows were dropped -- expected under 1%. "
        "A mass-drop like this previously indicated a timestamp-parsing bug."
    )

    assert set(metadata["label_counts"].keys()) <= EXPECTED_LABELS
    assert set(metadata["label_counts"].keys()) & ZERO_DAY_LABELS == ZERO_DAY_LABELS
    assert metadata["zero_day_holdout_row_count"] > 0
    assert 60 <= metadata["num_features"] <= 78
    assert set(metadata["day_counts"].keys()) == {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    }
