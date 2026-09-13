"""E6's "Ours+DP+SecAgg" comparator: a FIXED public clip bound (not
Phase 8's per-round adaptive median, which needs server visibility into
individual client norms that SecAgg+ specifically hides) applied
client-side, uniform SecAgg+ weighting via num_examples=1, and
DPSecAggPersonalizedFedAvg noising the revealed mean delta before
reconstructing the new global parameters. See DPSecAggPersonalizedFedAvg's
and run_dp_secagg_personalized_simulation's docstrings in simulation.py
for the full mechanism and clip_bound's real-data-anchored default.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.federated.simulation import DPSecAggPersonalizedFedAvg, run_dp_secagg_personalized_simulation

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


# TEST: aggregate_fit's reconstruction math, isolated from SecAgg+/Ray --
# noise_multiplier=0.0 (the "no DP" sentinel, per add_gaussian_noise's own
# docstring) must reconstruct EXACTLY old_global + revealed_mean_delta.
def test_dp_secagg_aggregate_fit_reconstructs_global_params_noise_free():
    class _FitRes:
        def __init__(self, arrays):
            from flwr.common import ndarrays_to_parameters
            self.parameters = ndarrays_to_parameters(arrays)
            self.num_examples = 1
            self.metrics = {}

    template = DPSecAggPersonalizedFedAvg.__new__(DPSecAggPersonalizedFedAvg)
    template.noise_multiplier = 0.0
    template.clip_bound = 5.0
    template.clients_per_round = 4
    template.rng = np.random.default_rng(0)
    template.current_global_params = [np.array([1.0, 2.0], dtype=np.float32)]
    template.accept_failures = True
    template.fit_metrics_aggregation_fn = None

    # SecAgg+ has already revealed the mean clipped delta -- every result
    # in `results` carries an IDENTICAL copy (see secaggplus_workflow.py),
    # so a single-entry `results` list is a faithful stand-in here.
    mean_delta = np.array([0.3, -0.1], dtype=np.float32)
    results = [(None, _FitRes([mean_delta]))]

    from flwr.common import parameters_to_ndarrays
    parameters, _ = template.aggregate_fit(server_round=1, results=results, failures=[])
    reconstructed = parameters_to_ndarrays(parameters)

    assert np.allclose(reconstructed[0], [1.3, 1.9])
    # the tracked baseline itself must have advanced too (next round's anchor)
    assert np.allclose(template.current_global_params[0], [1.3, 1.9])


# TEST: with noise, the reconstructed params equal old_global + noised
# mean_delta EXACTLY as add_gaussian_noise would independently compute it
# (i.e. the noise is calibrated against clip_bound / clients_per_round,
# an AVERAGE's sensitivity -- not clip_bound alone, which would be the
# SUM's sensitivity Phase 8's DPPersonalizedFedAvg uses instead).
def test_dp_secagg_aggregate_fit_noise_uses_average_sensitivity():
    class _FitRes:
        def __init__(self, arrays):
            from flwr.common import ndarrays_to_parameters
            self.parameters = ndarrays_to_parameters(arrays)
            self.num_examples = 1
            self.metrics = {}

    from fedpda_ids.privacy.dp import add_gaussian_noise

    clip_bound = 8.0
    clients_per_round = 4
    noise_multiplier = 1.5
    mean_delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    template = DPSecAggPersonalizedFedAvg.__new__(DPSecAggPersonalizedFedAvg)
    template.noise_multiplier = noise_multiplier
    template.clip_bound = clip_bound
    template.clients_per_round = clients_per_round
    template.rng = np.random.default_rng(123)
    template.current_global_params = [np.zeros(3, dtype=np.float32)]
    template.accept_failures = True
    template.fit_metrics_aggregation_fn = None

    results = [(None, _FitRes([mean_delta]))]
    from flwr.common import parameters_to_ndarrays
    parameters, _ = template.aggregate_fit(server_round=1, results=results, failures=[])
    reconstructed = parameters_to_ndarrays(parameters)[0]

    # Independently recompute what the SAME rng seed should have produced,
    # using the identical (average) sensitivity convention.
    expected_rng = np.random.default_rng(123)
    expected_noised = add_gaussian_noise([mean_delta], noise_multiplier, clip_bound / clients_per_round, expected_rng)
    assert np.allclose(reconstructed, expected_noised[0])


# TEST: end-to-end through the real Flower ServerApp/ClientApp/SecAgg+
# harness, target_epsilon=inf (noise_multiplier=0.0) so the run is
# deterministic given the seed -- exercises the full client-clips-a-
# fixed-bound + uniform-SecAgg+-weighting + strategy-reconstruction path.
def test_dp_secagg_personalized_simulation_end_to_end(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_dp_secagg_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="dp_secagg_smoke", seed=42,
        target_epsilon=float("inf"), target_delta=1e-5, clip_bound=5.0,
        clipping_range=16.0, max_weight=1000.0, modulus_range=2**30,
    )

    assert result["pool"] == [0, 1, 2]
    assert result["run_config"]["secagg_plus"] is True
    assert result["run_config"]["dp"] is True
    assert result["run_config"]["clip_bound"] == 5.0
    assert result["achieved_epsilon"] == float("inf")

    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["last_checkpoint"]).exists()

    ckpt = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    assert all(k.startswith(("encoder.", "decoder.")) for k in ckpt["shared_state_dict"].keys())

    summary = result["per_client_summary"]
    assert summary["num_clients_evaluated"] == 3


# TEST: a finite epsilon calibrates a nonzero noise_multiplier and still
# completes the full harness (real DP noise actually gets injected, not
# just the eps=inf sentinel path).
def test_dp_secagg_personalized_simulation_finite_epsilon(tmp_path):
    seq_dir = _build_synthetic_seq_dir(tmp_path, num_clients=3)

    result = run_dp_secagg_personalized_simulation(
        seq_dir=seq_dir, client_id_col="client_id", num_clients_configured=3,
        clients_per_round=3, num_rounds=2, local_epochs=1, batch_size=4,
        model_cfg=_model_cfg(), window_size=W, min_train_sequences=5, min_train_classes=2,
        checkpoint_dir=tmp_path / "checkpoints", run_name="dp_secagg_finite_eps", seed=42,
        target_epsilon=3.0, target_delta=1e-5, clip_bound=5.0,
        clipping_range=16.0, max_weight=1000.0, modulus_range=2**30,
    )

    assert result["run_config"]["noise_multiplier"] > 0.0
    assert np.isfinite(result["achieved_epsilon"])
    assert Path(result["last_checkpoint"]).exists()
