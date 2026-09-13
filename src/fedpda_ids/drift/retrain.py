"""E5's live drift-triggered retraining -- the piece Phase 10's own
kickoff decision deliberately deferred ("these functions DETECT and
REPORT where the trigger condition would fire; actually re-training on
trigger is deferred"). This module supplies the one new piece of
plumbing that decision left out: recovering the REAL chronological
cutoff a completed Phase 10 run's `first_trigger_round` corresponds to,
so a per-client retrain/eval split can be built at exactly that point
in time -- reusing Phase 10's own already-computed trigger round
rather than re-deriving a drift boundary from scratch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trigger_cutoff_timestamp(sorted_window_start_times: np.ndarray, round_window_size: int, first_trigger_round: int):
    """The real timestamp at the boundary between round
    `first_trigger_round - 1` and round `first_trigger_round` in the
    SAME pooled, chronologically-sorted stream detect_drift.py replayed
    (every real client's test-split sequences, globally sorted by
    window_start_time, chunked into `round_window_size`-sized rounds).

    Returned in `sorted_window_start_times`'s OWN dtype (never coerced
    to float) -- CICIDS2017's real window_start_time is a pandas
    datetime64 column, and forcing a float cast here made the later
    datetime64-vs-float comparison in split_client_rows_at_cutoff raise
    a real TypeError (found via the first real-data run, not a
    synthetic test -- synthetic fixtures elsewhere in this project use
    plain int/float "time" columns, which never exposed this).

    `first_trigger_round` must not be None -- callers should not call
    this for a scope where Phase 10's trigger never fired at all."""
    if first_trigger_round is None:
        raise ValueError("first_trigger_round is None -- the retrain trigger never fired for this scope.")
    cutoff_position = first_trigger_round * round_window_size
    if cutoff_position >= len(sorted_window_start_times):
        raise ValueError(
            f"first_trigger_round*round_window_size={cutoff_position} falls past the end of the "
            f"stream (length {len(sorted_window_start_times)})."
        )
    return sorted_window_start_times[cutoff_position]


def split_client_rows_at_cutoff(client_rows: pd.DataFrame, time_col: str, cutoff_timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """retrain_rows = strictly BEFORE the cutoff (data this client would
    really have collected by the time the trigger fired -- what a live
    retrain has available to use), eval_rows = AT/AFTER the cutoff
    (held out, never retrained on -- used to measure whether retraining
    actually helped)."""
    retrain_rows = client_rows[client_rows[time_col] < cutoff_timestamp]
    eval_rows = client_rows[client_rows[time_col] >= cutoff_timestamp]
    return retrain_rows, eval_rows
