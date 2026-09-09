"""Dataset-agnostic preprocessing helpers shared by CICIDS2017 and N-BaIoT.

Both federations go through the same shape of pipeline (clean -> split
chronologically per traffic source -> fit transforms on train only ->
partition into federated clients), so the mechanics live here once and
each dataset module (cicids2017.py, nbaiot.py) only supplies the
dataset-specific schema, label rules, and file loading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def coerce_numeric_and_flag_bad_rows(
    df: pd.DataFrame, feature_cols: list[str], copy: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Force feature columns to numeric, turn +/-Inf into NaN, and return
    a boolean mask of rows that are fully clean (no NaN in any feature).

    CICIDS2017 and N-BaIoT both contain rows with literal "Infinity"
    strings (from flows with ~0 duration, causing bytes/s = x/0) and
    occasional garbled numeric fields. We do not try to impute these --
    per the reproducibility rules, we drop and report the exact count
    rather than silently guessing values that would bias later stats.

    `copy=False` mutates `df` in place instead of copying it first --
    N-BaIoT's ~7M-row, 115-feature table is large enough that an extra
    full copy here risks exhausting memory on a 16GB machine; pass
    `copy=False` only when the caller doesn't need the original df.
    """
    if copy:
        df = df.copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    clean_mask = df[feature_cols].notna().all(axis=1)
    return df, clean_mask


def assign_chronological_split(
    df: pd.DataFrame,
    group_col: str | list[str],
    time_col: str,
    train_fraction: float,
    val_fraction: float,
) -> pd.Series:
    """Per group (traffic source), sort by time and label the earliest
    `train_fraction` of rows "train", the next `val_fraction` "val", and
    the remainder "test".

    This is done per group, not globally, so that every group has its
    own train/val/test tail -- which is what makes it safe to later
    build sliding-window sequences per group without a window ever
    straddling a split boundary.

    `group_col` accepts a list of columns (grouping on the tuple of
    their values) instead of forcing the caller to first materialize a
    single concatenated string key -- avoids an extra large object-dtype
    column on datasets where memory is already tight.
    """
    split = pd.Series(index=df.index, dtype="object")

    for _, group_idx in df.groupby(group_col).groups.items():
        ordered = df.loc[group_idx, time_col].sort_values().index
        n = len(ordered)
        n_train = int(np.floor(n * train_fraction))
        n_val = int(np.floor(n * val_fraction))

        split.loc[ordered[:n_train]] = "train"
        split.loc[ordered[n_train : n_train + n_val]] = "val"
        split.loc[ordered[n_train + n_val :]] = "test"

    return split


def select_log1p_columns(
    train_features: pd.DataFrame, skew_threshold: float
) -> list[str]:
    """Pick non-negative, heavily right-skewed columns for a log1p
    transform, decided from the TRAIN split only (leakage-safe).

    Skew is computed in float64 even when the column itself is float32:
    skew's central-moment cubing step can overflow float32's ~3.4e38
    range for columns with very large values (seen in N-BaIoT's
    HH_jit_*_variance features, up to ~1e17), silently producing NaN --
    which then fails `> skew_threshold` and skips log1p for exactly the
    columns that need it most. float64 has enough headroom (~1.8e308)
    that this doesn't happen for any value this project's data reaches.
    Casting one column at a time is cheap (one Series, not the whole
    feature block), so this doesn't reintroduce the memory problem the
    float32 dtype was chosen to avoid.
    """
    selected = []
    for col in train_features.columns:
        series = train_features[col]
        if (series >= 0).all() and series.astype("float64").skew() > skew_threshold:
            selected.append(col)
    return selected


def drop_constant_columns(
    train_features: pd.DataFrame, min_unique: int = 2
) -> list[str]:
    """Return feature columns to KEEP -- i.e. those with at least
    `min_unique` distinct values in the TRAIN split. Constant columns
    carry no information and can destabilize scaling."""
    return [
        col
        for col in train_features.columns
        if train_features[col].nunique(dropna=True) >= min_unique
    ]


def fit_minmax_scaler(train_features: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_features.values)
    return scaler


def dirichlet_partition_with_splits(
    labels: pd.Series,
    split: pd.Series,
    num_clients: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Assign each row to one of `num_clients` federated clients using
    label-skewed Dirichlet partitioning (Hsu et al., 2019), applied
    consistently across the train/val/test splits.

    For each class we draw ONE Dirichlet proportion vector over clients
    and reuse it for that class's rows in train, val, and test alike --
    so a client's label skew is the same shape in all three splits,
    only the row counts differ. Rows with a NaN label (e.g. zero-day
    holdout rows, which the caller excludes before calling this) never
    reach this function.

    Lower alpha -> more skewed (more non-IID) client label distributions.
    alpha -> infinity would approach a uniform (IID) split.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(labels.dropna().unique())
    proportions = {
        c: rng.dirichlet(np.repeat(alpha, num_clients)) for c in classes
    }

    client_id = np.full(len(labels), -1, dtype=int)
    labels_arr = labels.to_numpy()
    split_arr = split.to_numpy()
    positions = np.arange(len(labels))

    for split_value in pd.unique(split_arr):
        for c in classes:
            mask = (split_arr == split_value) & (labels_arr == c)
            idx = positions[mask]
            rng.shuffle(idx)

            props = proportions[c]
            cut_points = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            groups = np.split(idx, cut_points)

            for cid, group in enumerate(groups):
                client_id[group] = cid

    return client_id


def assert_no_chronological_leakage(
    df: pd.DataFrame, group_col: str | list[str], time_col: str, split_col: str = "temporal_split"
) -> None:
    """Raise AssertionError if any group has a train row later than a
    val row, or a val row later than a test row.

    This is the one property every processed dataset in this project
    must satisfy before Phase 3 builds sequences from it -- used by
    both datasets' test suites (Part B, Part D) and by the Part E
    cross-dataset check, so the leakage definition itself only exists
    once instead of two near-identical copies drifting apart.
    """
    for _, group in df.groupby(group_col):
        train_t = group.loc[group[split_col] == "train", time_col]
        val_t = group.loc[group[split_col] == "val", time_col]
        test_t = group.loc[group[split_col] == "test", time_col]

        if len(train_t) and len(val_t):
            assert train_t.max() <= val_t.min(), "train/val leakage detected"
        if len(val_t) and len(test_t):
            assert val_t.max() <= test_t.min(), "val/test leakage detected"
