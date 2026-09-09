"""Phase 3: genuine temporal sequence construction.

Turns Phase 2's flow-level processed tables into (window_size, F)
LSTM-ready sequences. See the Phase 3 design writeup for the two
findings that shape this module:

1. Grouping by a dataset's base entity column alone (Source IP, or
   device+Label+shard) is NOT sufficient to build leakage-safe
   sequences. Client assignment for CICIDS2017 is row-level Dirichlet,
   not host-level, so a single host's rows can span many different
   clients (verified: 77% of hosts span >1 client under some alpha).
   And a group's chronological train/val/test split cuts *within* its
   own timeline, so a naive window can straddle that cut too. The fix:
   group by (base columns + temporal_split + the active client-id
   column) -- not base columns alone. This also happens to isolate
   zero-day rows automatically, since "zero_day_holdout" is one of
   temporal_split's values and is therefore already a distinct group.

2. Label policy: a window can span rows with different labels
   (expected and common for CICIDS2017, since one host's traffic
   naturally transitions between BENIGN and attack states within one
   continuous capture). This module labels a sequence with its LAST
   row's label -- matching the "classify the current moment given
   recent context" framing the whole architecture is built around,
   and critically, avoiding majority-vote's bias toward erasing rare
   classes from their own windows. `is_label_pure` is recorded per
   sequence so this choice is inspectable later, not hidden.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("fedpda_ids")


def compute_window_count(n_rows: int, window_size: int, stride: int) -> int:
    """Number of stride-spaced, non-overlapping-start windows of length
    window_size that fit in n_rows chronologically ordered rows."""
    if n_rows < window_size:
        return 0
    return (n_rows - window_size) // stride + 1


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    base_group_cols: list[str],
    time_col: str,
    label_col: str,
    client_id_col: str,
    window_size: int,
    stride: int,
    output_dir: str | Path,
    storage_dtype: type = np.float32,
) -> dict:
    """Build (window_size, F) sequences from df, grouped by
    (base_group_cols + ["temporal_split", client_id_col]) and sorted by
    time_col within each group.

    Written directly to a disk-backed memmap (not accumulated in RAM
    then concatenated) -- N-BaIoT's sequence tensor can reach several
    GB, and this machine's Phase 2 experience showed that an extra
    full-size copy at the wrong moment is the difference between
    fitting in 16GB and not.

    `storage_dtype` defaults to float32 but can be set to float16 to
    halve disk usage under real storage pressure (this machine's C:
    drive hit 96%+ full during Phase 3). Values are already MinMax-
    scaled to roughly [0,1] by Phase 2, so float16's ~3 decimal digits
    of precision costs nothing that matters here; whatever loads X.npy
    for training should upcast to float32 first, since that's what the
    frozen architecture expects -- this is a storage-layer choice only,
    not a model precision change.

    Returns a summary dict (also written to output_dir/summary.json).
    Sequence-level metadata (label, client id, split, group key,
    window start/end time, is_label_pure) is written to
    output_dir/metadata.parquet, one row per sequence, in the same
    order as the rows of output_dir/X.npy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = [*base_group_cols, "temporal_split", client_id_col]
    grouped = df.groupby(group_cols, sort=False, observed=True)

    # Pass 1: window counts only depend on group SIZE, not row order, so
    # this is cheap (no per-row feature access) and tells us the total
    # sequence count up front -- needed to size the memmap before Pass 2.
    sizes = grouped.size()
    window_counts = sizes.apply(lambda n: compute_window_count(n, window_size, stride))
    total_sequences = int(window_counts.sum())
    num_eligible_groups = int((window_counts > 0).sum())
    num_ineligible_groups = int((window_counts == 0).sum())

    logger.info(
        "%s groups total: %d eligible (>=%d rows), %d ineligible (<%d rows) -> %d sequences",
        output_dir.name, num_eligible_groups, window_size, num_ineligible_groups, window_size, total_sequences,
    )

    num_features = len(feature_cols)
    x_path = output_dir / "X.npy"

    if total_sequences == 0:
        summary = {
            "output_dir": str(output_dir),
            "window_size": window_size,
            "stride": stride,
            "num_features": num_features,
            "group_cols": group_cols,
            "num_eligible_groups": 0,
            "num_ineligible_groups": num_ineligible_groups,
            "total_sequences": 0,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary

    # Retry the memmap allocation a few times: on this machine, disk free
    # space has been observed to fluctuate in real time (something else
    # on the system transiently consumes space), causing "No space left
    # on device" errors that then succeed if retried moments later even
    # though `df`/Get-PSDrive reported unchanged free space throughout.
    # Not a substitute for having genuine headroom -- just resilience
    # against a few-second dip.
    last_error: OSError | None = None
    X = None
    for attempt in range(5):
        try:
            X = np.lib.format.open_memmap(
                x_path, mode="w+", dtype=storage_dtype, shape=(total_sequences, window_size, num_features)
            )
            break
        except OSError as e:
            last_error = e
            logger.warning(
                "Attempt %d/5 to allocate %s failed (%s); retrying in %ds",
                attempt + 1, x_path, e, 2 ** attempt,
            )
            time.sleep(2**attempt)
    if X is None:
        raise RuntimeError(f"Failed to allocate {x_path} after 5 attempts") from last_error

    metadata_rows = []
    offset = 0

    # NOTE: iterating `grouped` directly (`for key, group_df in grouped:`)
    # triggers pandas' internal _sorted_data step, which gathers the
    # ENTIRE dataframe into one group-contiguous block before yielding
    # the first group -- effectively a full extra copy of the feature
    # block. On N-BaIoT's ~3GB (float32) table this alone caused a
    # numpy.core._exceptions._ArrayMemoryError. `.groups.items()` gives
    # {key: row_index} cheaply (no data movement), and `.loc[]` per key
    # extracts only that small group -- the same pattern nbaiot.py's
    # _assign_shards() already uses successfully on this dataset.
    for group_key, group_index in grouped.groups.items():
        n = len(group_index)
        n_windows = compute_window_count(n, window_size, stride)
        if n_windows == 0:
            continue

        ordered = df.loc[group_index].sort_values(time_col)
        feat_values = ordered[feature_cols].to_numpy(dtype=storage_dtype)
        label_values = ordered[label_col].to_numpy()
        time_values = ordered[time_col].to_numpy()

        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        key_dict = dict(zip(group_cols, key_tuple))

        for w in range(n_windows):
            start = w * stride
            end = start + window_size

            X[offset] = feat_values[start:end]
            window_labels = label_values[start:end]
            seq_label = window_labels[-1]
            is_pure = bool((window_labels == seq_label).all())

            metadata_rows.append(
                {
                    **key_dict,
                    "sequence_label": seq_label,
                    "is_label_pure": is_pure,
                    "window_start_time": time_values[start],
                    "window_end_time": time_values[end - 1],
                    "sequence_index": offset,
                }
            )
            offset += 1

    X.flush()
    del X  # release the memmap handle

    assert offset == total_sequences, f"wrote {offset} sequences, expected {total_sequences}"

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_parquet(output_dir / "metadata.parquet", index=False)

    summary = {
        "output_dir": str(output_dir),
        "window_size": window_size,
        "stride": stride,
        "num_features": num_features,
        "storage_dtype": np.dtype(storage_dtype).name,
        "group_cols": group_cols,
        "client_id_col": client_id_col,
        "num_eligible_groups": num_eligible_groups,
        "num_ineligible_groups": num_ineligible_groups,
        "total_sequences": total_sequences,
        "label_counts": metadata_df["sequence_label"].value_counts().to_dict(),
        "split_counts": metadata_df["temporal_split"].value_counts().to_dict(),
        "sequences_per_client": metadata_df.loc[
            metadata_df[client_id_col] >= 0, client_id_col
        ].value_counts().to_dict(),
        "impure_sequence_count": int((~metadata_df["is_label_pure"]).sum()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
