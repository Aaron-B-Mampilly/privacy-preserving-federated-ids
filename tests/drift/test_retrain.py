"""E5's live drift-triggered retraining -- unit tests for the small
timestamp-cutoff helpers (trigger_cutoff_timestamp/split_client_rows_at_cutoff).
The full retrain loop (run_drift_triggered_retraining) is tested
end-to-end in tests/federated/test_drift_retraining_simulation.py.
"""

import numpy as np
import pandas as pd
import pytest

from fedpda_ids.drift.retrain import split_client_rows_at_cutoff, trigger_cutoff_timestamp


def test_trigger_cutoff_timestamp_picks_the_right_boundary():
    # 10 rounds of 3 sequences each, times 0..29
    times = np.arange(30, dtype=float)
    cutoff = trigger_cutoff_timestamp(times, round_window_size=3, first_trigger_round=2)
    assert cutoff == 6.0  # round 2 starts at position 2*3=6


def test_trigger_cutoff_timestamp_none_raises():
    with pytest.raises(ValueError):
        trigger_cutoff_timestamp(np.arange(10, dtype=float), round_window_size=3, first_trigger_round=None)


def test_trigger_cutoff_timestamp_past_end_raises():
    with pytest.raises(ValueError):
        trigger_cutoff_timestamp(np.arange(10, dtype=float), round_window_size=3, first_trigger_round=10)


def test_split_client_rows_at_cutoff_partitions_correctly():
    df = pd.DataFrame({"window_start_time": [1, 2, 3, 4, 5, 6], "x": list(range(6))})
    retrain, eval_ = split_client_rows_at_cutoff(df, "window_start_time", cutoff_timestamp=4.0)

    assert list(retrain["window_start_time"]) == [1, 2, 3]
    assert list(eval_["window_start_time"]) == [4, 5, 6]
    assert len(retrain) + len(eval_) == len(df)


def test_split_client_rows_at_cutoff_empty_retrain_or_eval():
    df = pd.DataFrame({"window_start_time": [10, 11, 12], "x": [0, 1, 2]})

    all_retrain, none_eval = split_client_rows_at_cutoff(df, "window_start_time", cutoff_timestamp=100.0)
    assert len(all_retrain) == 3 and len(none_eval) == 0

    none_retrain, all_eval = split_client_rows_at_cutoff(df, "window_start_time", cutoff_timestamp=0.0)
    assert len(none_retrain) == 0 and len(all_eval) == 3
