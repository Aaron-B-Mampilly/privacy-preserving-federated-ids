"""Phase 4: PyTorch Dataset/DataLoader over Phase 3 sequence artifacts.

Reads `metadata.parquet` (small -- bookkeeping columns only) to decide
which rows a given (split, client) scope needs, then fancy-indexes
only those rows out of the `X.npy` memmap -- never loads the whole
tensor into RAM. N-BaIoT's X.npy is float16 on disk (a Phase 3 disk-
space decision); every sample is upcast to float32 in __getitem__,
before it ever reaches the model, per the frozen training design.

Class-index mapping policy (Phase 4 Part A finding): a training
scope's classes are exactly the labels PRESENT IN THAT SCOPE'S OWN
TRAIN SPLIT -- not the dataset's full label set. This is what
correctly excludes CICIDS2017's DDoS (zero training sequences at any
alpha, any client -- a structural consequence of the frozen
chronological drift design, not a bug) and what correctly handles a
local client whose train split happens to be missing a class its own
val/test split has. Any label encountered at eval time that isn't in
the scope's label_to_index is never coerced into some existing class
-- it's counted and reported separately (see `unseen_eval_labels`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("fedpda_ids")


def build_label_index(labels) -> dict[str, int]:
    """Deterministic (sorted) label -> integer index mapping."""
    return {label: i for i, label in enumerate(sorted(set(labels)))}


def _read_metadata(seq_dir: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(seq_dir / "metadata.parquet", columns=columns)


def scope_mask(metadata: pd.DataFrame, split: str, client_id_col: str | None, client_id) -> pd.Series:
    """True for rows belonging to (split, client) -- client_id=None means
    every non-holdout client (the centralized scope)."""
    mask = metadata["temporal_split"] == split
    if client_id_col is not None and client_id is not None:
        mask &= metadata[client_id_col] == client_id
    elif client_id_col is not None and client_id is None and split != "zero_day_holdout":
        # centralized: every real client, never the -1 holdout sentinel
        mask &= metadata[client_id_col] != -1
    return mask


def get_scope_train_labels(seq_dir: Path, client_id_col: str | None, client_id) -> list[str]:
    """Labels actually present in this scope's TRAIN split -- the basis
    for that scope's class-index mapping (see module docstring)."""
    cols = ["temporal_split", "sequence_label"] + ([client_id_col] if client_id_col else [])
    metadata = _read_metadata(seq_dir, cols)
    mask = scope_mask(metadata, "train", client_id_col, client_id)
    return sorted(metadata.loc[mask, "sequence_label"].unique())


def check_client_trainable(
    seq_dir: Path, client_id_col: str, client_id, min_train_sequences: int, min_train_classes: int
) -> tuple[bool, str]:
    """Policy check (Phase 4 design decision): a client is only trained
    if its own train split clears both bars. Returns (ok, reason)."""
    cols = ["temporal_split", "sequence_label", client_id_col]
    metadata = _read_metadata(seq_dir, cols)
    mask = scope_mask(metadata, "train", client_id_col, client_id)
    n = int(mask.sum())
    n_classes = metadata.loc[mask, "sequence_label"].nunique()

    if n < min_train_sequences:
        return False, f"insufficient data: {n} train sequences (< {min_train_sequences})"
    if n_classes < min_train_classes:
        return False, f"insufficient data: {n_classes} distinct train classes (< {min_train_classes})"
    return True, f"ok: {n} train sequences, {n_classes} classes"


def report_unseen_eval_labels(
    seq_dir: Path, split: str, client_id_col: str | None, client_id, label_to_index: dict[str, int]
) -> dict[str, int]:
    """Labels present in this (split, client) scope that are NOT in the
    scope's own label_to_index (built from its train split) -- e.g.
    CICIDS2017's DDoS, or a rare class a local client never trained on.
    Counted and returned, never silently folded into an existing class."""
    cols = ["temporal_split", "sequence_label"] + ([client_id_col] if client_id_col else [])
    metadata = _read_metadata(seq_dir, cols)
    mask = scope_mask(metadata, split, client_id_col, client_id)
    subset = metadata.loc[mask, "sequence_label"]
    unseen = subset[~subset.isin(label_to_index.keys())]
    return unseen.value_counts().to_dict()


class SequenceDataset(Dataset):
    """One (split, client) scope's sequences. `client_id=None` means
    every real client (the centralized scope); `split="zero_day_holdout"`
    is only ever meaningful with client_id=None (holdout rows are
    client_id=-1 for every scheme, per Phase 2/3)."""

    def __init__(
        self,
        seq_dir: str | Path,
        split: str,
        label_to_index: dict[str, int],
        client_id_col: str | None = None,
        client_id=None,
    ):
        self.seq_dir = Path(seq_dir)
        self.label_to_index = label_to_index

        cols = ["temporal_split", "sequence_label", "sequence_index"] + (
            [client_id_col] if client_id_col else []
        )
        metadata = _read_metadata(self.seq_dir, cols)
        mask = scope_mask(metadata, split, client_id_col, client_id)
        subset = metadata.loc[mask]

        # Drop rows whose label isn't in this scope's class mapping --
        # tracked via report_unseen_eval_labels(), not silently included
        # with a fabricated/incorrect class index.
        known = subset["sequence_label"].isin(label_to_index.keys())
        subset = subset.loc[known]

        self.sequence_indices = subset["sequence_index"].to_numpy()
        self.labels = subset["sequence_label"].to_numpy()
        self._X = None  # lazily opened -- safe whether num_workers is 0 or >0

    def _ensure_open(self):
        if self._X is None:
            self._X = np.load(self.seq_dir / "X.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        row_idx = self.sequence_indices[i]
        x = np.array(self._X[row_idx], dtype=np.float32)  # copies -- always writable, and float16 -> float32 upcast happens here
        y = self.label_to_index[self.labels[i]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_scope_dataloaders(
    seq_dir: str | Path,
    client_id_col: str | None,
    client_id,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    label_to_index: dict[str, int] | None = None,
) -> dict:
    """Builds train/val/test DataLoaders (+ zero-day holdout dataset) for
    one scope (centralized if client_id is None, else one client), plus
    the scope's label_to_index and any unseen-eval-label report.

    Does not touch the zero_day_holdout split's client boundary the
    same way as train/val/test: holdout rows are always client_id=-1,
    shared across the whole dataset, evaluated once per experiment
    rather than per client -- callers typically build it once at
    centralized scope and reuse it.

    `label_to_index`, if given, OVERRIDES the default "derive classes
    from this scope's own train split" behavior (Phase 4's policy) --
    used by Phase 5's FedAvg, where every client must share one global
    class vocabulary for the classifier head to be weight-averageable
    at all (see federated/ module docstring for why this isn't optional).
    """
    seq_dir = Path(seq_dir)
    if label_to_index is None:
        train_labels = get_scope_train_labels(seq_dir, client_id_col, client_id)
        label_to_index = build_label_index(train_labels)

    datasets = {
        split: SequenceDataset(seq_dir, split, label_to_index, client_id_col, client_id)
        for split in ("train", "val", "test")
    }

    unseen = {
        split: report_unseen_eval_labels(seq_dir, split, client_id_col, client_id, label_to_index)
        for split in ("val", "test")
    }
    for split, counts in unseen.items():
        if counts:
            logger.warning(
                "scope client_id=%s: %s split has labels absent from this scope's train set "
                "(excluded from standard metrics, reported separately): %s",
                client_id, split, counts,
            )

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
    }

    return {
        "loaders": loaders,
        "datasets": datasets,
        "label_to_index": label_to_index,
        "index_to_label": {i: label for label, i in label_to_index.items()},
        "unseen_eval_labels": unseen,
    }


class ZeroDaySequenceDataset(Dataset):
    """Zero-day holdout sequences, for reconstruction-MSE evaluation and
    composition reporting ONLY -- not classification.

    Zero-day labels are by definition never in any scope's train-
    derived label_to_index (that's what makes them zero-day), so
    SequenceDataset's "filter to known labels" behavior would always
    return an empty dataset here. Phase 4 explicitly does not attempt
    zero-day classification (that's Phase 7's prototype mechanism) --
    this class returns (x, label_string) instead of (x, class_index),
    so callers can compute reconstruction MSE and report label
    composition without a fabricated class target.
    """

    def __init__(self, seq_dir: str | Path):
        self.seq_dir = Path(seq_dir)
        metadata = _read_metadata(self.seq_dir, ["temporal_split", "sequence_label", "sequence_index"])
        subset = metadata.loc[metadata["temporal_split"] == "zero_day_holdout"]
        self.sequence_indices = subset["sequence_index"].to_numpy()
        self.labels = subset["sequence_label"].to_numpy()
        self._X = None

    def _ensure_open(self):
        if self._X is None:
            self._X = np.load(self.seq_dir / "X.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        self._ensure_open()
        row_idx = self.sequence_indices[i]
        x = np.array(self._X[row_idx], dtype=np.float32)  # copies -- always writable
        return torch.from_numpy(x), str(self.labels[i])


class SequenceIndexDataset(Dataset):
    """An arbitrary, caller-supplied set of sequences by explicit
    `sequence_index` -- for subsets that don't fit SequenceDataset's
    (split, client) scope model, e.g. Phase 10's drift-detection
    reference set (last N chronological BENIGN windows) or its
    monitored stream (a scope's test split re-ordered chronologically).
    Order is preserved exactly as given -- callers needing chronological
    order must sort `sequence_indices` themselves before constructing
    this. Returns (x, dummy_label=0) since drift detection is
    unsupervised and never needs a real class target."""

    def __init__(self, seq_dir: str | Path, sequence_indices):
        self.seq_dir = Path(seq_dir)
        self.sequence_indices = np.asarray(sequence_indices)
        self._X = None

    def _ensure_open(self):
        if self._X is None:
            self._X = np.load(self.seq_dir / "X.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        row_idx = self.sequence_indices[i]
        x = np.array(self._X[row_idx], dtype=np.float32)  # copies -- always writable
        return torch.from_numpy(x), torch.tensor(0, dtype=torch.long)
