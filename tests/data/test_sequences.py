"""Phase 3 Part D: sequence-construction tests.

The fixture is deliberately adversarial, not just a happy path:
- "H1"/train/client=0 and "H1"/train/client=1 share the SAME host and
  the SAME time values (1..15) but disjoint feature-value ranges
  (1-15 vs 101-115) -- a real client-boundary bug (grouping by host
  only) would interleave them when sorted by time, which is exactly
  what this is built to catch.
- "H1"/val/client=0 shares the host with the train groups above but a
  disjoint value range (301-310) -- catches a split-boundary bug.
- "H2"/train/client=0 has only 8 rows (< window_size) -- must produce
  zero sequences.
- "H3"/zero_day_holdout/client=-1 (value range 401-415) -- must stay
  isolated and never blend into any train/val group.

Every group's value range is disjoint from every other's, so "does
any window mix two ranges" is a simple, strong boundary-violation check.
"""

import numpy as np
import pandas as pd
import pytest

from fedpda_ids.data.sequences import build_sequences, compute_window_count

FEATURE_COLS = ["value", "value_x10"]


def _rows(host, split, client, values, labels):
    n = len(values)
    return pd.DataFrame(
        {
            "host": [host] * n,
            "temporal_split": [split] * n,
            "client_id": [client] * n,
            "time": values,
            "value": values,
            "value_x10": [v * 10 for v in values],
            "Label": labels,
        }
    )


@pytest.fixture
def synthetic_df():
    h1_client0_labels = ["BENIGN"] * 7 + ["ATTACK"] * 3 + ["BENIGN"] * 5  # values 1..15
    frames = [
        _rows("H1", "train", 0, list(range(1, 16)), h1_client0_labels),
        _rows("H1", "train", 1, list(range(101, 116)), ["BENIGN"] * 15),
        _rows("H2", "train", 0, list(range(201, 209)), ["BENIGN"] * 8),  # only 8 rows
        _rows("H1", "val", 0, list(range(301, 311)), ["BENIGN"] * 10),  # exactly 10
        _rows("H3", "zero_day_holdout", -1, list(range(401, 416)), ["ZERO_DAY"] * 15),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def built(tmp_path, synthetic_df):
    summary = build_sequences(
        synthetic_df,
        feature_cols=FEATURE_COLS,
        base_group_cols=["host"],
        time_col="time",
        label_col="Label",
        client_id_col="client_id",
        window_size=10,
        stride=5,
        output_dir=tmp_path / "seqs",
    )
    X = np.load(tmp_path / "seqs" / "X.npy")
    metadata = pd.read_parquet(tmp_path / "seqs" / "metadata.parquet")
    return X, metadata, summary


# ---------------------------------------------------------------------
# TEST 1-2: shape
# ---------------------------------------------------------------------


def test_window_length_is_exactly_10(built):
    X, _, _ = built
    assert X.shape[1] == 10


def test_feature_dimension_matches_schema(built):
    X, _, _ = built
    assert X.shape[2] == len(FEATURE_COLS)


# ---------------------------------------------------------------------
# TEST 3: stride
# ---------------------------------------------------------------------


def test_stride_is_exactly_5(built):
    _, metadata, _ = built
    h1_train0 = metadata[(metadata["host"] == "H1") & (metadata["temporal_split"] == "train") & (metadata["client_id"] == 0)]
    h1_train0 = h1_train0.sort_values("sequence_index")
    starts = h1_train0["window_start_time"].to_numpy()
    assert len(starts) == 2
    assert starts[1] - starts[0] == 5


# ---------------------------------------------------------------------
# TEST 4: monotonic order within a sequence
# ---------------------------------------------------------------------


def test_temporal_order_is_monotonically_increasing(built):
    X, _, _ = built
    value_feature = X[:, :, FEATURE_COLS.index("value")]
    diffs = np.diff(value_feature, axis=1)
    assert (diffs > 0).all()


# ---------------------------------------------------------------------
# TEST 5: adjacent-window overlap
# ---------------------------------------------------------------------


def test_adjacent_windows_overlap_by_exactly_5(built):
    X, metadata, _ = built
    h1_train0 = metadata[(metadata["host"] == "H1") & (metadata["temporal_split"] == "train") & (metadata["client_id"] == 0)]
    h1_train0 = h1_train0.sort_values("sequence_index")
    idx0, idx1 = h1_train0["sequence_index"].to_numpy()

    window0_values = X[idx0, :, FEATURE_COLS.index("value")]
    window1_values = X[idx1, :, FEATURE_COLS.index("value")]

    assert np.array_equal(window0_values[5:], window1_values[:5])


# ---------------------------------------------------------------------
# TEST 6-8: boundaries (group / client / split)
# ---------------------------------------------------------------------

# Disjoint value ranges per (host, split, client) group in the fixture.
_EXPECTED_RANGES = {
    ("H1", "train", 0): (1, 15),
    ("H1", "train", 1): (101, 115),
    ("H2", "train", 0): (201, 208),
    ("H1", "val", 0): (301, 310),
    ("H3", "zero_day_holdout", -1): (401, 415),
}


def test_no_sequence_crosses_a_group_client_or_split_boundary(built):
    X, metadata, _ = built
    value_feature = X[:, :, FEATURE_COLS.index("value")]

    for _, row in metadata.iterrows():
        key = (row["host"], row["temporal_split"], row["client_id"])
        lo, hi = _EXPECTED_RANGES[key]
        window_values = value_feature[row["sequence_index"]]
        assert window_values.min() >= lo and window_values.max() <= hi, (
            f"sequence {row['sequence_index']} (key={key}) has values outside its "
            f"group's range [{lo},{hi}]: {window_values}"
        )


def test_short_group_produces_no_sequences(built):
    _, metadata, _ = built
    h2 = metadata[metadata["host"] == "H2"]
    assert len(h2) == 0


# ---------------------------------------------------------------------
# TEST 9: zero-day isolation
# ---------------------------------------------------------------------


def test_zero_day_sequences_are_isolated(built):
    _, metadata, _ = built
    zero_day = metadata[metadata["temporal_split"] == "zero_day_holdout"]
    assert len(zero_day) > 0
    assert (zero_day["client_id"] == -1).all()
    assert (zero_day["sequence_label"] == "ZERO_DAY").all()
    # and no non-zero-day sequence should ever carry client_id == -1
    non_zero_day = metadata[metadata["temporal_split"] != "zero_day_holdout"]
    assert (non_zero_day["client_id"] != -1).all()


# ---------------------------------------------------------------------
# TEST 10: no identifier leaks into features
# ---------------------------------------------------------------------


def test_no_identifier_columns_in_feature_list():
    identifier_like = {"host", "time", "client_id", "temporal_split", "Label"}
    assert identifier_like.isdisjoint(set(FEATURE_COLS))


# ---------------------------------------------------------------------
# TEST 11: finiteness
# ---------------------------------------------------------------------


def test_no_nan_or_inf_introduced(built):
    X, _, _ = built
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------
# TEST 12: sequence-to-client mapping
# ---------------------------------------------------------------------


def test_sequence_to_client_mapping_is_correct(built):
    X, metadata, _ = built
    value_feature = X[:, :, FEATURE_COLS.index("value")]

    for _, row in metadata.iterrows():
        window_values = value_feature[row["sequence_index"]]
        if row["client_id"] == 1:
            assert window_values.min() >= 101
        elif row["client_id"] == 0 and row["temporal_split"] == "train":
            assert window_values.max() <= 15
        elif row["client_id"] == -1:
            assert window_values.min() >= 401


# ---------------------------------------------------------------------
# TEST 13-14: determinism
# ---------------------------------------------------------------------


def test_sequence_counts_deterministic_across_runs(tmp_path, synthetic_df):
    kwargs = dict(
        df=synthetic_df,
        feature_cols=FEATURE_COLS,
        base_group_cols=["host"],
        time_col="time",
        label_col="Label",
        client_id_col="client_id",
        window_size=10,
        stride=5,
    )
    summary_a = build_sequences(output_dir=tmp_path / "run_a", **kwargs)
    summary_b = build_sequences(output_dir=tmp_path / "run_b", **kwargs)

    assert summary_a["total_sequences"] == summary_b["total_sequences"]
    X_a = np.load(tmp_path / "run_a" / "X.npy")
    X_b = np.load(tmp_path / "run_b" / "X.npy")
    assert np.array_equal(X_a, X_b)


def test_determinism_unaffected_by_global_random_seed(tmp_path, synthetic_df):
    kwargs = dict(
        df=synthetic_df,
        feature_cols=FEATURE_COLS,
        base_group_cols=["host"],
        time_col="time",
        label_col="Label",
        client_id_col="client_id",
        window_size=10,
        stride=5,
    )
    np.random.seed(1)
    summary_a = build_sequences(output_dir=tmp_path / "seed1", **kwargs)
    np.random.seed(999)
    summary_b = build_sequences(output_dir=tmp_path / "seed999", **kwargs)

    assert summary_a["total_sequences"] == summary_b["total_sequences"]
    X_a = np.load(tmp_path / "seed1" / "X.npy")
    X_b = np.load(tmp_path / "seed999" / "X.npy")
    assert np.array_equal(X_a, X_b)


# ---------------------------------------------------------------------
# TEST 15: literal worked example from the spec
# ---------------------------------------------------------------------


def test_literal_worked_example_produces_exact_expected_windows():
    df = pd.DataFrame(
        {
            "host": ["H"] * 15,
            "temporal_split": ["train"] * 15,
            "client_id": [0] * 15,
            "time": list(range(1, 16)),
            "value": list(range(1, 16)),
            "Label": ["BENIGN"] * 15,
        }
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        summary = build_sequences(
            df,
            feature_cols=["value"],
            base_group_cols=["host"],
            time_col="time",
            label_col="Label",
            client_id_col="client_id",
            window_size=10,
            stride=5,
            output_dir=tmp,
        )
        assert summary["total_sequences"] == 2
        X = np.load(f"{tmp}/X.npy")

        expected_window_1 = np.arange(1, 11).reshape(10, 1)
        expected_window_2 = np.arange(6, 16).reshape(10, 1)
        forbidden_window = np.arange(2, 12).reshape(10, 1)  # stride=1 would produce this

        windows = [X[0], X[1]]
        assert any(np.array_equal(w, expected_window_1) for w in windows)
        assert any(np.array_equal(w, expected_window_2) for w in windows)
        assert not any(np.array_equal(w, forbidden_window) for w in windows)


# ---------------------------------------------------------------------
# compute_window_count unit checks
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_rows,window_size,stride,expected",
    [
        (15, 10, 5, 2),
        (10, 10, 5, 1),
        (9, 10, 5, 0),
        (14, 10, 5, 1),
        (20, 10, 5, 3),
    ],
)
def test_compute_window_count(n_rows, window_size, stride, expected):
    assert compute_window_count(n_rows, window_size, stride) == expected
