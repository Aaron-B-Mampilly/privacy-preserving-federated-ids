"""Phase 9: end-to-end SecAgg+-personalized-FL simulation test, through
the real Flower ServerApp/ClientApp/run_simulation harness (not mocked,
not the legacy start_simulation API Phases 5/6/8 use) -- mirrors
test_personalized_simulation.py's synthetic setup.
"""

from pathlib import Path

import pandas as pd
import torch

from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.federated.simulation import run_secagg_personalized_simulation

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


# TEST: synthetic SecAgg+ run completes through the real Flower harness
def test_secagg_personalized_simulation_end_to_end(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_secagg_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="secagg_smoke", seed=42,
        clipping_range=16.0, max_weight=1000.0, modulus_range=2**30,
    )

    assert result["pool"] == [0, 1, 2]
    assert len(result["history_losses_centralized"]) == 3  # round 0 (initial) + 2 rounds
    assert result["best_round"] >= 0
    assert result["best_val_mse"] < float("inf")

    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["last_checkpoint"]).exists()

    # checkpoint must contain ONLY shared (encoder/decoder) keys, same
    # contract as Phase 6's plain personalized FL -- SecAgg+ doesn't
    # change WHAT gets communicated, only HOW it's aggregated.
    ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert all(k.startswith(("encoder.", "decoder.")) for k in ckpt["shared_state_dict"].keys())
    assert not any(k.startswith("classifier.") for k in ckpt["shared_state_dict"].keys())

    summary = result["per_client_summary"]
    assert summary["num_clients_evaluated"] == 3
    assert result["run_config"]["secagg_plus"] is True


# TEST: per-client boundaries remain correct under SecAgg+ (same
# disk-persisted-head mechanism as plain personalized FL -- SecAgg+
# only changes shared-parameter aggregation, never touches local heads)
def test_secagg_personalized_local_heads_remain_independent(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_secagg_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="secagg_heads", seed=42,
        clipping_range=16.0, max_weight=1000.0, modulus_range=2**30,
    )

    heads_dir = tmp_path / "checkpoints" / "personalized_heads" / "secagg_heads"
    head_files = sorted(heads_dir.glob("client_*.pt"))
    assert len(head_files) == 3

    heads = [torch.load(p, weights_only=True) for p in head_files]
    assert any(not torch.equal(heads[0][k], heads[1][k]) for k in heads[0])
    assert result["per_client_summary"]["num_clients_skipped_no_head"] == 0
