"""Phase 2 Part E: common data-pipeline validation.

Both CICIDS2017 and N-BaIoT go through dataset-specific pipelines
(different raw schemas, different client structures -- Dirichlet vs.
fixed device/shard), but Phase 3 needs to build sequences from
*either* dataset without special-casing. This file checks that both
processed outputs actually satisfy one shared contract, using the
config's declared per-dataset sequence_group_columns/
sequence_time_column rather than hardcoding dataset-specific column
names here.

The contract, concretely:
  - a "Label" column (same name and dtype in both)
  - an "is_zero_day_holdout" boolean column
  - a "temporal_split" column with only {"train","val","test","zero_day_holdout"}
  - at least one client-id column, -1 for every holdout row, a valid
    non-negative id for every other row
  - no chronological leakage when grouped/sorted per the dataset's own
    declared sequence_group_columns / sequence_time_column

These are checked twice: once against the fast synthetic fixtures
(every run) and once against the real metadata + a light read of the
real parquet's bookkeeping columns (skipped if you haven't run the
real preprocessing yet).
"""

from pathlib import Path

import pandas as pd
import pytest

from fedpda_ids.data.common import assert_no_chronological_leakage
from fedpda_ids.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_SPLIT_VALUES = {"train", "val", "test", "zero_day_holdout"}


def _check_common_contract(df: pd.DataFrame, client_id_cols: list[str], group_col, time_col: str) -> None:
    assert "Label" in df.columns
    assert df["Label"].dtype == object or str(df["Label"].dtype).startswith("string")

    assert "is_zero_day_holdout" in df.columns
    assert df["is_zero_day_holdout"].dtype == bool

    assert "temporal_split" in df.columns
    assert set(df["temporal_split"].unique()) <= ALLOWED_SPLIT_VALUES

    holdout = df[df["is_zero_day_holdout"]]
    non_holdout = df[~df["is_zero_day_holdout"]]
    assert (holdout["temporal_split"] == "zero_day_holdout").all()
    assert (non_holdout["temporal_split"] != "zero_day_holdout").all()

    for col in client_id_cols:
        assert col in df.columns
        assert df[col].dtype.kind == "i"
        assert (holdout[col] == -1).all()
        assert (non_holdout[col] >= 0).all()

    assert_no_chronological_leakage(non_holdout, group_col=group_col, time_col=time_col)


# ---------------------------------------------------------------------
# Synthetic (every run)
# ---------------------------------------------------------------------


def test_cicids2017_satisfies_common_contract(processed_cicids):
    df, metadata = processed_cicids
    _check_common_contract(
        df,
        client_id_cols=metadata["client_id_columns"],
        group_col="Source IP",
        time_col="Timestamp",
    )


def test_nbaiot_satisfies_common_contract(processed_nbaiot):
    df, _ = processed_nbaiot
    _check_common_contract(
        df,
        client_id_cols=["client_id_45", "client_id_9"],
        group_col=["device", "Label", "shard_id"],
        time_col="row_order",
    )


# ---------------------------------------------------------------------
# Real data (skipped if either hasn't been preprocessed yet)
# ---------------------------------------------------------------------

CICIDS_PROCESSED = REPO_ROOT / "data" / "processed" / "cicids2017" / "full_processed.parquet"
NBAIOT_PROCESSED = REPO_ROOT / "data" / "processed" / "nbaiot" / "full_processed.parquet"


@pytest.mark.skipif(
    not (CICIDS_PROCESSED.exists() and NBAIOT_PROCESSED.exists()),
    reason="both real datasets must be preprocessed first (run scripts/preprocess_cicids2017.py and preprocess_nbaiot.py)",
)
def test_real_datasets_both_satisfy_common_contract():
    config = load_config(REPO_ROOT / "configs" / "config.yaml")

    cic_cfg = config["data"]["cicids2017"]
    cic_bookkeeping = [
        "Label", "is_zero_day_holdout", "temporal_split",
        *cic_cfg["sequence_group_columns"], cic_cfg["sequence_time_column"],
        "client_id_alpha5", "client_id_alpha0.5", "client_id_alpha0.1",
    ]
    cic_df = pd.read_parquet(CICIDS_PROCESSED, columns=sorted(set(cic_bookkeeping)))
    _check_common_contract(
        cic_df,
        client_id_cols=["client_id_alpha5", "client_id_alpha0.5", "client_id_alpha0.1"],
        group_col=cic_cfg["sequence_group_columns"],
        time_col=cic_cfg["sequence_time_column"],
    )

    nbaiot_cfg = config["data"]["nbaiot"]
    nbaiot_bookkeeping = [
        "Label", "is_zero_day_holdout", "temporal_split",
        *nbaiot_cfg["sequence_group_columns"], nbaiot_cfg["sequence_time_column"],
        "client_id_45", "client_id_9",
    ]
    nbaiot_df = pd.read_parquet(NBAIOT_PROCESSED, columns=sorted(set(nbaiot_bookkeeping)))
    _check_common_contract(
        nbaiot_df,
        client_id_cols=["client_id_45", "client_id_9"],
        group_col=nbaiot_cfg["sequence_group_columns"],
        time_col=nbaiot_cfg["sequence_time_column"],
    )
