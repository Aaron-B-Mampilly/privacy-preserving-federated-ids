"""Phase 5: end-to-end FedAvg simulation test, through the real Flower/
Ray harness (not mocked) -- on tiny synthetic data so it stays fast.
"""

import pandas as pd
import torch

from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.federated.simulation import build_trainable_client_pool, run_fedavg_simulation

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
    labels_cycle = (["BENIGN"] * 6 + ["ATTACK"] * 6) * 3  # 3 classes' worth of coverage per client
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


def test_trainable_pool_matches_synthetic_clients(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)
    pool = build_trainable_client_pool(seq_dir, "client_id", 3, min_train_sequences=5, min_train_classes=2)
    assert pool == [0, 1, 2]


def test_fedavg_simulation_end_to_end(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)
    model_cfg = {
        "latent_dim": 32,
        "encoder": {"layer1_units": 8},
        "decoder": {"layer1_units": 8},
        "classifier_head": {"hidden_units": 8, "dropout": 0.2},
        "loss": {"lambda_ce": 1.0},
        "optimizer": {"learning_rate": 1e-3},
    }

    result = run_fedavg_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=model_cfg, window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="fedavg_smoke", seed=42,
    )

    assert result["pool"] == [0, 1, 2]
    assert len(result["history_losses_centralized"]) == 3  # round 0 (initial) + 2 rounds
    assert result["best_round"] >= 1
    assert result["best_val_loss"] < float("inf")

    import numpy as np
    assert np.isfinite(result["test_metrics"]["accuracy"])
    assert np.isfinite(result["test_metrics"]["macro_f1"])

    from pathlib import Path
    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["last_checkpoint"]).exists()

    ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["run_config"]["num_clients_pool"] == 3


# TEST (E1's FedProx comparator): runs end-to-end through the real
# Flower/Ray harness with proximal_mu > 0 -- same server-side
# aggregation as plain FedAvg, just a different client-side loss.
def test_fedprox_simulation_end_to_end(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)
    model_cfg = {
        "latent_dim": 32,
        "encoder": {"layer1_units": 8},
        "decoder": {"layer1_units": 8},
        "classifier_head": {"hidden_units": 8, "dropout": 0.2},
        "loss": {"lambda_ce": 1.0},
        "optimizer": {"learning_rate": 1e-3},
    }

    result = run_fedavg_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=model_cfg, window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="fedprox_smoke", seed=42,
        proximal_mu=0.01,
    )

    assert result["pool"] == [0, 1, 2]
    assert result["run_config"]["proximal_mu"] == 0.01

    import numpy as np
    assert np.isfinite(result["test_metrics"]["accuracy"])
    assert np.isfinite(result["test_metrics"]["macro_f1"])
