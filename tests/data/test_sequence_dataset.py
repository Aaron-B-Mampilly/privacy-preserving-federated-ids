"""Phase 4 Part D: Dataset/DataLoader + class-index mapping tests.

Validated on a synthetic sequence artifact (built via the real Phase 3
build_sequences(), not hand-rolled) BEFORE any real-data training, per
explicit instruction. Covers Tests 7, 9, 11, 12 plus the two Phase 4
Part A design decisions (DDoS-style train-absent labels, extreme
non-IID client skip policy) and the float16->float32 upcast.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from fedpda_ids.data.sequence_dataset import (
    LabeledSequenceIndexDataset,
    SequenceIndexDataset,
    ZeroDaySequenceDataset,
    build_label_index,
    build_scope_dataloaders,
    check_client_trainable,
    get_scope_train_labels,
    report_unseen_eval_labels,
)
from fedpda_ids.data.sequences import build_sequences

FEATURE_COLS = ["value"]


def _rows(host, split, client, values, labels):
    n = len(values)
    return pd.DataFrame(
        {
            "host": [host] * n,
            "temporal_split": [split] * n,
            "client_id": [client] * n,
            "time": values,
            "value": values,
            "Label": labels,
        }
    )


@pytest.fixture
def seq_dir(tmp_path):
    """A synthetic dataset engineered to exercise every Part D case:
    - client 0: normal train/val/test, 2 classes (BENIGN, ATTACK)
    - client 1: DDoS-style bug -- has val/test ATTACK2 rows but ZERO
      train rows of that class (mirrors CICIDS2017's real DDoS finding)
    - client 2: extreme non-IID -- only 1 train sequence, 1 class ->
      must be skipped by check_client_trainable, never trained
    - zero-day holdout rows (client_id=-1) for a class never in any
      client's train set
    """
    frames = [
        _rows("H", "train", 0, list(range(0, 30)), ["BENIGN"] * 20 + ["ATTACK"] * 10),
        _rows("H", "val", 0, list(range(100, 115)), ["BENIGN"] * 15),
        _rows("H", "test", 0, list(range(200, 215)), ["BENIGN"] * 15),
        _rows("H", "train", 1, list(range(300, 320)), ["BENIGN"] * 20),
        _rows("H", "val", 1, list(range(400, 415)), ["ATTACK2"] * 15),  # never in client 1's train
        _rows("H", "test", 1, list(range(500, 515)), ["ATTACK2"] * 15),
        # exactly 10 raw rows -> exactly 1 sequence, matching the real
        # extreme-non-IID case found in Phase 3 (CICIDS2017 alpha=0.1,
        # client 18: 1 train sequence, 1 class)
        _rows("H", "train", 2, list(range(600, 610)), ["BENIGN"] * 10),
        _rows("H", "zero_day_holdout", -1, list(range(700, 720)), ["ZERO_DAY"] * 20),
    ]
    df = pd.concat(frames, ignore_index=True)

    output_dir = tmp_path / "seqs"
    build_sequences(
        df,
        feature_cols=FEATURE_COLS,
        base_group_cols=["host"],
        time_col="time",
        label_col="Label",
        client_id_col="client_id",
        window_size=10,
        stride=5,
        output_dir=output_dir,
    )
    return output_dir


# ---------------------------------------------------------------------
# Class-index mapping
# ---------------------------------------------------------------------


def test_build_label_index_is_deterministic():
    idx_a = build_label_index(["b", "a", "c"])
    idx_b = build_label_index(["c", "b", "a"])
    assert idx_a == idx_b == {"a": 0, "b": 1, "c": 2}


def test_centralized_scope_train_labels_exclude_never_trained_classes(seq_dir):
    # ATTACK2 only appears in client 1's val/test, never in ANY client's
    # train split -- centralized scope must not include it either.
    labels = get_scope_train_labels(seq_dir, client_id_col="client_id", client_id=None)
    assert "ATTACK2" not in labels
    assert set(labels) == {"BENIGN", "ATTACK"}


def test_ddos_style_unseen_label_is_reported_not_absorbed(seq_dir):
    """Client 1's val/test ATTACK2 rows must be excluded from its
    dataset (not silently relabeled) and reported separately."""
    train_labels = get_scope_train_labels(seq_dir, client_id_col="client_id", client_id=1)
    assert train_labels == ["BENIGN"]  # client 1 never saw ATTACK2 in train
    label_to_index = build_label_index(train_labels)

    unseen_val = report_unseen_eval_labels(seq_dir, "val", "client_id", 1, label_to_index)
    unseen_test = report_unseen_eval_labels(seq_dir, "test", "client_id", 1, label_to_index)
    assert unseen_val.get("ATTACK2", 0) > 0
    assert unseen_test.get("ATTACK2", 0) > 0


# ---------------------------------------------------------------------
# Extreme non-IID client skip policy
# ---------------------------------------------------------------------


def test_client_with_one_sequence_is_flagged_not_trainable(seq_dir):
    ok, reason = check_client_trainable(
        seq_dir, client_id_col="client_id", client_id=2, min_train_sequences=10, min_train_classes=2
    )
    assert ok is False
    assert "insufficient data" in reason


def test_normal_client_is_flagged_trainable(seq_dir):
    ok, reason = check_client_trainable(
        seq_dir, client_id_col="client_id", client_id=0, min_train_sequences=1, min_train_classes=2
    )
    assert ok is True


def test_single_class_client_fails_min_classes_bar(seq_dir):
    # client 1 has plenty of train sequences but only 1 class (BENIGN)
    ok, reason = check_client_trainable(
        seq_dir, client_id_col="client_id", client_id=1, min_train_sequences=1, min_train_classes=2
    )
    assert ok is False
    assert "classes" in reason


# ---------------------------------------------------------------------
# TEST 11: client boundary -- a client's dataset must never contain
# another client's sequences
# ---------------------------------------------------------------------


def test_local_dataset_cannot_contain_another_clients_sequences(seq_dir):
    scope0 = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=0, batch_size=4)
    scope1 = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=1, batch_size=4)

    # client 0's train values are all < 100 (see fixture); client 1's are >= 300
    for x, _ in scope0["loaders"]["train"]:
        assert (x[:, :, 0] < 100).all()
    for x, _ in scope1["loaders"]["train"]:
        assert (x[:, :, 0] >= 300).all()


# ---------------------------------------------------------------------
# TEST 12: test/zero-day rows never reachable from the TRAIN DataLoader
# ---------------------------------------------------------------------


def test_train_loader_never_yields_test_or_zero_day_rows(seq_dir):
    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=0, batch_size=4)
    train_values = set()
    for x, _ in scope["loaders"]["train"]:
        train_values.update(x[:, :, 0].flatten().tolist())

    # client 0's train range is [0,29); val is [100,115); test is [200,215);
    # zero-day is [700,720) -- none of those should appear via the train loader
    assert all(v < 30 for v in train_values)


def test_zero_day_dataset_is_separate_and_never_in_train_loader(seq_dir):
    zero_day = ZeroDaySequenceDataset(seq_dir)
    assert len(zero_day) > 0
    for i in range(len(zero_day)):
        x, label = zero_day[i]
        assert label == "ZERO_DAY"
        assert (x[:, 0] >= 700).all()

    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=None, batch_size=4)
    for x, _ in scope["loaders"]["train"]:
        assert (x[:, :, 0] < 700).all()


# ---------------------------------------------------------------------
# float16 -> float32 upcast (N-BaIoT's actual storage dtype)
# ---------------------------------------------------------------------


def test_float16_storage_upcasts_to_float32_before_reaching_model(tmp_path, seq_dir):
    # Rebuild the same fixture with float16 storage (mirrors N-BaIoT).
    df = pd.concat(
        [
            _rows("H", "train", 0, list(range(0, 30)), ["BENIGN"] * 20 + ["ATTACK"] * 10),
            _rows("H", "val", 0, list(range(100, 115)), ["BENIGN"] * 15),
            _rows("H", "test", 0, list(range(200, 215)), ["BENIGN"] * 15),
        ],
        ignore_index=True,
    )
    output_dir = tmp_path / "seqs_f16"
    build_sequences(
        df,
        feature_cols=FEATURE_COLS,
        base_group_cols=["host"],
        time_col="time",
        label_col="Label",
        client_id_col="client_id",
        window_size=10,
        stride=5,
        output_dir=output_dir,
        storage_dtype=np.float16,
    )

    on_disk = np.load(output_dir / "X.npy", mmap_mode="r")
    assert on_disk.dtype == np.float16

    scope = build_scope_dataloaders(output_dir, client_id_col="client_id", client_id=0, batch_size=4)
    for x, y in scope["loaders"]["train"]:
        assert x.dtype == torch.float32
        assert y.dtype == torch.long


# ---------------------------------------------------------------------
# TEST 9 (dataset level): incomplete final batch
# ---------------------------------------------------------------------


def test_incomplete_final_batch_from_real_dataloader(seq_dir):
    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=0, batch_size=1000)
    batches = list(scope["loaders"]["train"])
    assert len(batches) == 1  # everything fits in one (necessarily incomplete) batch
    x, y = batches[0]
    assert x.shape[0] == y.shape[0] == len(scope["datasets"]["train"])


# ---------------------------------------------------------------------
# Phase 10: SequenceIndexDataset -- arbitrary explicit-index subsets
# (drift detection's reference set / chronologically-reordered stream)
# ---------------------------------------------------------------------


def test_sequence_index_dataset_preserves_given_order(seq_dir):
    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=0, batch_size=1000)
    all_indices = scope["datasets"]["train"].sequence_indices
    assert len(all_indices) >= 2

    reversed_indices = all_indices[::-1]
    dataset = SequenceIndexDataset(seq_dir, reversed_indices)
    assert len(dataset) == len(reversed_indices)

    x_direct = np.load(seq_dir / "X.npy", mmap_mode="r")
    for i, expected_row in enumerate(reversed_indices):
        x, y = dataset[i]
        assert np.allclose(x.numpy(), x_direct[expected_row].astype(np.float32))
        assert y.item() == 0  # dummy label -- unsupervised, never a real class target


def test_sequence_index_dataset_empty():
    dataset = SequenceIndexDataset("unused", [])
    assert len(dataset) == 0


# ---------------------------------------------------------------------
# E5's live drift-triggered retraining: LabeledSequenceIndexDataset --
# same "arbitrary explicit-index subset" idea as SequenceIndexDataset,
# but with REAL class targets (needed for local retraining's classifier
# loss, unlike Phase 10's unsupervised-only drift monitoring).
# ---------------------------------------------------------------------


def test_labeled_sequence_index_dataset_returns_real_class_targets(seq_dir):
    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=0, batch_size=1000)
    train_ds = scope["datasets"]["train"]
    label_to_index = scope["label_to_index"]

    dataset = LabeledSequenceIndexDataset(seq_dir, train_ds.sequence_indices, train_ds.labels, label_to_index)

    assert len(dataset) == len(train_ds)
    x_direct = np.load(seq_dir / "X.npy", mmap_mode="r")
    for i in range(len(dataset)):
        x, y = dataset[i]
        assert np.allclose(x.numpy(), x_direct[train_ds.sequence_indices[i]].astype(np.float32))
        assert y.item() == label_to_index[train_ds.labels[i]]


def test_labeled_sequence_index_dataset_drops_unknown_labels(seq_dir):
    # client 1's val split is entirely ATTACK2, which is absent from
    # client 1's own train-derived label_to_index (the DDoS-style case) --
    # must be dropped, never fabricated into an existing class index.
    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=1, batch_size=1000)
    val_ds = scope["datasets"]["val"]  # already filtered by SequenceDataset itself
    label_to_index = scope["label_to_index"]  # {"BENIGN": 0} -- client 1's train never saw ATTACK2

    # Read the RAW (unfiltered) val metadata directly to prove the drop is real.
    metadata = pd.read_parquet(
        seq_dir / "metadata.parquet", columns=["temporal_split", "sequence_label", "sequence_index", "client_id"],
    )
    raw_val = metadata[(metadata["temporal_split"] == "val") & (metadata["client_id"] == 1)]
    assert len(raw_val) > 0
    assert set(raw_val["sequence_label"].unique()) == {"ATTACK2"}

    dataset = LabeledSequenceIndexDataset(
        seq_dir, raw_val["sequence_index"].to_numpy(), raw_val["sequence_label"].to_numpy(), label_to_index,
    )
    assert len(dataset) == 0
    assert len(val_ds) == 0  # consistent with SequenceDataset's own filtering


def test_labeled_sequence_index_dataset_handles_already_empty_input():
    # A genuinely EMPTY input (zero rows to begin with, e.g. a client with
    # no data on one side of a chronological cutoff) is a different case
    # from "some rows present but all filtered out" above: np.array([])
    # on an empty Python list defaults to float64, which then fails as a
    # boolean mask -- a real bug found via a real N-BaIoT drift-retrain run.
    dataset = LabeledSequenceIndexDataset(
        "unused", np.array([], dtype=np.int64), np.array([], dtype=object), {"BENIGN": 0},
    )
    assert len(dataset) == 0
