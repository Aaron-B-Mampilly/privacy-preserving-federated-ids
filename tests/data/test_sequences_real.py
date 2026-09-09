"""Phase 3 Parts G, I, J: real-data and cross-dataset sequence validation.

Skipped entirely if the real sequence artifacts haven't been built yet
(run scripts/build_sequences_cicids2017.py / build_sequences_nbaiot.py
first). Uses memmap for X.npy so this stays memory-light regardless
of dataset size -- N-BaIoT's tensor is ~3GB even at float16.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

CICIDS_SEQ_DIRS = {
    alpha: REPO_ROOT / "data" / "processed" / "cicids2017" / "sequences" / f"alpha_{alpha}"
    for alpha in ["5", "0.5", "0.1"]
}
NBAIOT_SEQ_DIRS = {
    "main_45": REPO_ROOT / "data" / "processed" / "nbaiot" / "sequences" / "main_45",
}

ALL_SEQ_DIRS = {**{f"cicids2017/alpha_{k}": v for k, v in CICIDS_SEQ_DIRS.items()},
                **{f"nbaiot/{k}": v for k, v in NBAIOT_SEQ_DIRS.items()}}


def _built(seq_dir: Path) -> bool:
    return (seq_dir / "X.npy").exists() and (seq_dir / "metadata.parquet").exists() and (seq_dir / "summary.json").exists()


def _load(seq_dir: Path):
    summary = json.loads((seq_dir / "summary.json").read_text())
    metadata = pd.read_parquet(seq_dir / "metadata.parquet")
    X = np.load(seq_dir / "X.npy", mmap_mode="r")
    return X, metadata, summary


@pytest.mark.parametrize("name,seq_dir", list(ALL_SEQ_DIRS.items()))
def test_real_sequence_shape_and_counts_are_self_consistent(name, seq_dir):
    if not _built(seq_dir):
        pytest.skip(f"{name} not built yet")
    X, metadata, summary = _load(seq_dir)

    assert X.shape[0] == len(metadata) == summary["total_sequences"]
    assert X.shape[1] == summary["window_size"] == 10
    assert X.shape[2] == summary["num_features"]


@pytest.mark.parametrize("name,seq_dir", list(ALL_SEQ_DIRS.items()))
def test_real_sequences_are_all_finite(name, seq_dir):
    if not _built(seq_dir):
        pytest.skip(f"{name} not built yet")
    X, _, _ = _load(seq_dir)

    chunk = 50_000
    for start in range(0, X.shape[0], chunk):
        block = np.asarray(X[start : start + chunk])
        assert np.isfinite(block).all(), f"{name}: non-finite values in rows [{start}:{start+chunk}]"


@pytest.mark.parametrize(
    "name,seq_dir,client_id_col",
    [(f"cicids2017/alpha_{a}", d, f"client_id_alpha{a}") for a, d in CICIDS_SEQ_DIRS.items()]
    + [(f"nbaiot/{s}", d, "client_id_45") for s, d in NBAIOT_SEQ_DIRS.items()],
)
def test_real_sequence_zero_day_isolation_and_client_validity(name, seq_dir, client_id_col):
    if not _built(seq_dir):
        pytest.skip(f"{name} not built yet")
    _, metadata, _ = _load(seq_dir)

    holdout = metadata[metadata["temporal_split"] == "zero_day_holdout"]
    non_holdout = metadata[metadata["temporal_split"] != "zero_day_holdout"]

    assert (holdout[client_id_col] == -1).all()
    assert (non_holdout[client_id_col] >= 0).all()


# ---------------------------------------------------------------------
# Part J: cross-dataset common contract for sequence artifacts
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,seq_dir", list(ALL_SEQ_DIRS.items()))
def test_sequence_artifact_common_contract(name, seq_dir):
    """Every built sequence artifact -- regardless of dataset -- must
    expose the same three files with the same core metadata columns,
    so Phase 4 can load either dataset without special-casing."""
    if not _built(seq_dir):
        pytest.skip(f"{name} not built yet")
    _, metadata, summary = _load(seq_dir)

    for required_key in ("window_size", "stride", "num_features", "total_sequences", "group_cols"):
        assert required_key in summary

    for required_col in ("sequence_label", "is_label_pure", "temporal_split", "sequence_index"):
        assert required_col in metadata.columns

    assert summary["window_size"] == 10
    assert summary["stride"] == 5
