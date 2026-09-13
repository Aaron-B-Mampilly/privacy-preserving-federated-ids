"""E5's live drift-triggered retraining: run_drift_triggered_retraining
starts from a REAL completed personalized-FL checkpoint (built here via
the same run_personalized_simulation every other Phase 6+ test uses),
then retrains on synthetic "post-drift" data via a manual (non-Flower)
loop -- see its docstring in simulation.py for why no Ray harness is
needed here.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedpda_ids.data.sequence_dataset import LabeledSequenceIndexDataset, build_scope_dataloaders
from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.federated.simulation import load_shared_checkpoint, make_model, run_drift_triggered_retraining, run_personalized_simulation

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
        # "test" split doubles as the post-drift stream this experiment retrains/evaluates on
        frames.append(_rows("H", "test", client, list(range(base + 400, base + 460)), (labels_cycle * 2)[:60]))
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


def test_drift_triggered_retraining_changes_shared_params_and_saves_new_heads(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)
    checkpoint_dir = tmp_path / "checkpoints"
    source_run_name = "personalized_source"

    base_result = run_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=checkpoint_dir, run_name=source_run_name, seed=42,
    )
    pool = base_result["pool"]
    label_to_index = base_result["run_config"]["label_to_index"]

    # Build "retrain" DataLoaders directly from each client's own TEST-split
    # sequences (standing in for E5's "newly available post-drift data").
    metadata = pd.read_parquet(seq_dir / "metadata.parquet", columns=["temporal_split", "sequence_label", "sequence_index", "client_id"])
    retrain_loaders = {}
    for client_id in pool:
        client_test_rows = metadata[(metadata["temporal_split"] == "test") & (metadata["client_id"] == client_id)]
        dataset = LabeledSequenceIndexDataset(
            seq_dir, client_test_rows["sequence_index"].to_numpy(), client_test_rows["sequence_label"].to_numpy(), label_to_index,
        )
        retrain_loaders[client_id] = DataLoader(dataset, batch_size=4, shuffle=True)

    centralized_scope = build_scope_dataloaders(seq_dir, "client_id", None, 4, 0, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]
    num_classes = len(label_to_index)

    pre_model = make_model(num_features, num_classes, _model_cfg(), W)
    load_shared_checkpoint(checkpoint_dir / f"{source_run_name}_best.pt", pre_model)
    pre_shared_state = {k: v.clone() for k, v in pre_model.state_dict().items() if k.startswith(("encoder.", "decoder."))}

    result = run_drift_triggered_retraining(
        pool=pool, retrain_loaders=retrain_loaders, checkpoint_dir=checkpoint_dir,
        source_run_name=source_run_name, new_run_name="personalized_retrained",
        num_features=num_features, num_classes=num_classes, model_cfg=_model_cfg(), window_size=W,
        local_epochs=1, num_retrain_rounds=2, lambda_ce=1.0, learning_rate=1e-2, device=torch.device("cpu"),
    )

    assert result["num_retrain_rounds_run"] == 2
    assert set(result["clients_retrained"]) == set(pool)
    assert Path(result["new_shared_checkpoint"]).exists()

    post_model = make_model(num_features, num_classes, _model_cfg(), W)
    load_shared_checkpoint(Path(result["new_shared_checkpoint"]), post_model)
    post_shared_state = {k: v for k, v in post_model.state_dict().items() if k.startswith(("encoder.", "decoder."))}

    assert any(not torch.equal(pre_shared_state[k], post_shared_state[k]) for k in pre_shared_state)

    heads_dir = checkpoint_dir / "personalized_heads" / "personalized_retrained"
    assert len(list(heads_dir.glob("client_*.pt"))) == len(pool)

    # source run's own checkpoint/heads must be untouched (still loadable, unchanged)
    source_model = make_model(num_features, num_classes, _model_cfg(), W)
    load_shared_checkpoint(checkpoint_dir / f"{source_run_name}_best.pt", source_model)
    source_shared_state = {k: v for k, v in source_model.state_dict().items() if k.startswith(("encoder.", "decoder."))}
    assert all(torch.equal(pre_shared_state[k], source_shared_state[k]) for k in pre_shared_state)


def test_drift_triggered_retraining_stops_early_if_no_client_has_data(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)
    checkpoint_dir = tmp_path / "checkpoints"
    source_run_name = "personalized_source2"

    base_result = run_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=1, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=checkpoint_dir, run_name=source_run_name, seed=42,
    )
    pool = base_result["pool"]
    label_to_index = base_result["run_config"]["label_to_index"]
    centralized_scope = build_scope_dataloaders(seq_dir, "client_id", None, 4, 0, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]
    num_classes = len(label_to_index)

    result = run_drift_triggered_retraining(
        pool=pool, retrain_loaders={}, checkpoint_dir=checkpoint_dir,
        source_run_name=source_run_name, new_run_name="personalized_retrained_empty",
        num_features=num_features, num_classes=num_classes, model_cfg=_model_cfg(), window_size=W,
        local_epochs=1, num_retrain_rounds=3, lambda_ce=1.0, learning_rate=1e-2, device=torch.device("cpu"),
    )

    assert result["num_retrain_rounds_run"] == 0
    assert result["clients_retrained"] == []
