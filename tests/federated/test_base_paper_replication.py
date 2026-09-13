"""E1's "Base-paper replication" comparator: C. Sri Abhijit et al.'s FL
mechanism (uniform-averaged shared layers + N-round periodic transfer),
verified against the paper's open-access method description (PMC12484838),
not guessed. See BasePaperReplicationFedAvg's docstring for the full
reasoning and the one documented simplification (fixed period instead
of an adaptive significance threshold, since the paper's exact
threshold/metric isn't given in the available text).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.federated.simulation import BasePaperReplicationFedAvg, run_personalized_simulation

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


# TEST: uniform (1/K) averaging, not weighted by dataset size
def test_base_paper_replication_uses_uniform_not_weighted_averaging():
    class _FitRes:
        def __init__(self, arrays, num_examples):
            from flwr.common import ndarrays_to_parameters
            self.parameters = ndarrays_to_parameters(arrays)
            self.num_examples = num_examples
            self.metrics = {}

    template = BasePaperReplicationFedAvg.__new__(BasePaperReplicationFedAvg)
    template.transfer_period = 1  # every round transmits, isolating the averaging behavior
    template.current_shared_params = [np.zeros(3, dtype=np.float32)]
    template.accept_failures = True
    template.fit_metrics_aggregation_fn = None

    # client A: tiny dataset (10 examples), huge parameter value
    # client B: huge dataset (10000 examples), tiny parameter value
    # weighted-by-dataset-size would land close to B; uniform averaging lands exactly halfway.
    results = [
        (None, _FitRes([np.array([100.0, 100.0, 100.0], dtype=np.float32)], num_examples=10)),
        (None, _FitRes([np.array([0.0, 0.0, 0.0], dtype=np.float32)], num_examples=10000)),
    ]

    parameters, _ = template.aggregate_fit(server_round=1, results=results, failures=[])
    from flwr.common import parameters_to_ndarrays
    aggregated = parameters_to_ndarrays(parameters)
    assert np.allclose(aggregated[0], [50.0, 50.0, 50.0])  # exact midpoint, not weight-skewed toward B


# TEST: N-round periodic transfer -- global params only change on
# transmit rounds (server_round % transfer_period == 0)
def test_base_paper_replication_periodic_transfer_schedule():
    class _FitRes:
        def __init__(self, arrays):
            from flwr.common import ndarrays_to_parameters
            self.parameters = ndarrays_to_parameters(arrays)
            self.num_examples = 100
            self.metrics = {}

    template = BasePaperReplicationFedAvg.__new__(BasePaperReplicationFedAvg)
    template.transfer_period = 3
    template.current_shared_params = [np.array([1.0, 1.0], dtype=np.float32)]
    template.accept_failures = True
    template.fit_metrics_aggregation_fn = None

    results = [(None, _FitRes([np.array([9.0, 9.0], dtype=np.float32)]))]

    from flwr.common import parameters_to_ndarrays

    # round 1, 2: not multiples of 3 -- unchanged
    p1, _ = template.aggregate_fit(1, results, [])
    assert np.allclose(parameters_to_ndarrays(p1)[0], [1.0, 1.0])
    p2, _ = template.aggregate_fit(2, results, [])
    assert np.allclose(parameters_to_ndarrays(p2)[0], [1.0, 1.0])

    # round 3: a multiple of 3 -- transmits, updates to the client's value
    p3, _ = template.aggregate_fit(3, results, [])
    assert np.allclose(parameters_to_ndarrays(p3)[0], [9.0, 9.0])

    # round 4: back to holding steady at the new value
    p4, _ = template.aggregate_fit(4, results, [])
    assert np.allclose(parameters_to_ndarrays(p4)[0], [9.0, 9.0])


# TEST: end-to-end through the real Flower/Ray harness, reusing
# run_personalized_simulation's entire setup via strategy_cls/strategy_kwargs
def test_base_paper_replication_end_to_end(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=4, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="base_paper_smoke", seed=42,
        strategy_cls=BasePaperReplicationFedAvg, strategy_kwargs={"transfer_period": 3},
    )

    assert result["pool"] == [0, 1, 2]
    assert result["run_config"]["strategy_cls"] == "BasePaperReplicationFedAvg"
    assert Path(result["best_checkpoint"]).exists()

    summary = result["per_client_summary"]
    assert summary["num_clients_evaluated"] == 3
