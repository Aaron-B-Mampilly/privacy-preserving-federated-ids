"""Phase 8: end-to-end DP-personalized-FL simulation test, through the
real Flower/Ray harness (not mocked) -- mirrors test_personalized_simulation.py's
synthetic setup, adding assertions specific to the DP mechanism (adaptive
clipping, noise, achieved-epsilon reporting).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.federated.client import save_local_head
from fedpda_ids.federated.simulation import (
    evaluate_personalized_pool,
    make_model,
    run_dp_personalized_simulation,
    run_personalized_simulation,
    save_shared_checkpoint,
)

F = 6
W = 10


def _rows(host, split, client, values, labels):
    n = len(values)
    data = {"host": [host] * n, "temporal_split": [split] * n, "client_id": [client] * n, "time": values}
    for f in range(F):
        data[f"f{f}"] = [v + f for v in values]
    data["Label"] = labels
    return pd.DataFrame(data)


def _build_synthetic_seq_dir(tmp_path, num_clients=3):
    labels_cycle = (["BENIGN"] * 6 + ["ATTACK"] * 6) * 3
    frames = []
    for client in range(num_clients):
        base = client * 1000
        frames.append(_rows("H", "train", client, list(range(base, base + 60)), (labels_cycle * 2)[:60]))
        frames.append(_rows("H", "val", client, list(range(base + 200, base + 230)), labels_cycle[:30]))
        frames.append(_rows("H", "test", client, list(range(base + 400, base + 430)), labels_cycle[:30]))
    df = pd.concat(frames, ignore_index=True)

    output_dir = tmp_path / "seqs"
    build_sequences(
        df, feature_cols=[f"f{f}" for f in range(F)], base_group_cols=["host"],
        time_col="time", label_col="Label", client_id_col="client_id",
        window_size=W, stride=5, output_dir=output_dir,
    )
    return output_dir


def _model_cfg():
    return {
        "latent_dim": 32,
        "encoder": {"layer1_units": 8},
        "decoder": {"layer1_units": 8},
        "classifier_head": {"hidden_units": 8, "dropout": 0.2},
        "loss": {"lambda_ce": 1.0},
        "optimizer": {"learning_rate": 1e-3},
    }


# TEST: synthetic DP run completes, with real clipping+noise applied
def test_dp_personalized_simulation_end_to_end(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_dp_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="dp_smoke", seed=42,
        target_epsilon=3.0, target_delta=1e-5,
    )

    assert result["pool"] == [0, 1, 2]
    assert len(result["clip_norm_history"]) == 2  # one adaptive C_t per round
    assert all(c >= 0 for c in result["clip_norm_history"])
    assert result["run_config"]["noise_multiplier"] > 0.0
    assert np.isfinite(result["achieved_epsilon"])
    assert result["achieved_epsilon"] > 0

    assert Path(result["best_checkpoint"]).exists()
    ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert all(k.startswith(("encoder.", "decoder.")) for k in ckpt["shared_state_dict"].keys())

    summary = result["per_client_summary"]
    assert summary["num_clients_evaluated"] == 3


# TEST: target_epsilon=inf means noise_multiplier=0.0 (the "no DP" sentinel) --
# clipping still applies (a real, separate ablation from unclipped FedAvg).
def test_dp_personalized_simulation_infinite_epsilon_disables_noise(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_dp_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="dp_infinite", seed=42,
        target_epsilon=float("inf"), target_delta=1e-5,
    )

    assert result["run_config"]["noise_multiplier"] == 0.0
    assert result["achieved_epsilon"] == float("inf")
    assert len(result["clip_norm_history"]) == 2  # clipping is independent of noise


# TEST: stronger privacy (smaller epsilon) calibrates a larger noise_multiplier,
# reflected in run_config -- a basic sanity/monotonicity check on real wiring,
# not just the already-unit-tested dp.py formula.
def test_dp_personalized_simulation_smaller_epsilon_means_more_noise(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)
    kwargs = dict(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        seed=42, target_delta=1e-5,
    )
    result_loose = run_dp_personalized_simulation(
        checkpoint_dir=tmp_path / "loose", run_name="dp_loose", target_epsilon=8.0, **kwargs
    )
    result_tight = run_dp_personalized_simulation(
        checkpoint_dir=tmp_path / "tight", run_name="dp_tight", target_epsilon=0.5, **kwargs
    )
    assert result_tight["run_config"]["noise_multiplier"] > result_loose["run_config"]["noise_multiplier"]


# TEST: evaluate_personalized_pool's checkpoint_suffix param actually
# controls which file gets loaded -- the PRE-CODING CORRECTION for the
# real finding that "best" (lowest val MSE) always picks the untrained
# round-0 checkpoint under DP noise (confirmed on all 16 real sweep runs:
# e.g. N-BaIoT eps=8's val MSE went 0.141 (round 0) -> 0.320 (round 1),
# never recovering over 100 rounds). run_dp_personalized_simulation
# requests "last" specifically to avoid ever silently evaluating an
# untrained encoder.
def test_evaluate_personalized_pool_checkpoint_suffix_selects_the_right_file(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=1)
    checkpoint_dir = tmp_path / "checkpoints"
    run_name = "suffix_test"

    model = make_model(num_features=F, num_classes=2, model_cfg=_model_cfg(), window_size=W)
    save_shared_checkpoint(checkpoint_dir / f"{run_name}_last.pt", model, round_num=5, run_config={}, metrics={})
    save_local_head(model, checkpoint_dir / "personalized_heads" / run_name / "client_0.pt")
    # deliberately do NOT create f"{run_name}_best.pt"

    common = dict(
        pool=[0], seq_dir=seq_dir, client_id_col="client_id", checkpoint_dir=checkpoint_dir, run_name=run_name,
        num_features=F, num_classes=2, model_cfg=_model_cfg(), window_size=W, batch_size=4, num_workers=0,
        label_to_index={"BENIGN": 0, "ATTACK": 1}, index_to_label={0: "BENIGN", 1: "ATTACK"},
        lambda_ce=1.0, device=torch.device("cpu"),
    )

    # checkpoint_suffix="last" (what run_dp_personalized_simulation uses): succeeds
    metrics, summary = evaluate_personalized_pool(**common, checkpoint_suffix="last")
    assert summary["num_clients_evaluated"] == 1

    # default ("best", what run_personalized_simulation uses): no such file exists here
    import pytest
    with pytest.raises(FileNotFoundError):
        evaluate_personalized_pool(**common)
