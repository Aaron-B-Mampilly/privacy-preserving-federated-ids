"""Phase 10: ADWIN + PSI drift-detection mechanism tests, using
deliberately-constructed synthetic streams with a KNOWN shift point."""

import numpy as np
import torch

from fedpda_ids.drift.detectors import (
    compute_latent_psi,
    compute_per_sequence_errors_and_latents,
    compute_psi,
    evaluate_drift_stream,
    run_adwin_detector,
)


# ---------------------------------------------------------------------
# compute_per_sequence_errors_and_latents
# ---------------------------------------------------------------------


class _ConstantAutoencoder(torch.nn.Module):
    """Reconstructs input EXACTLY except for a fixed additive bias --
    lets tests control exact per-sequence MSE via that bias."""

    def __init__(self, bias: float, latent_dim: int = 4):
        super().__init__()
        self.bias = bias
        self.latent_dim = latent_dim

    def forward(self, x):
        reconstruction = x + self.bias
        latent = x[:, -1, : self.latent_dim]
        logits = torch.zeros(x.shape[0], 2)
        return reconstruction, logits, latent


def _make_loader(n, window=5, num_features=3, batch_size=8):
    x = torch.zeros(n, window, num_features)
    y = torch.zeros(n, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_compute_per_sequence_errors_matches_known_bias():
    loader = _make_loader(n=10, num_features=4)
    model = _ConstantAutoencoder(bias=2.0, latent_dim=4)  # reconstruction error = bias^2 everywhere
    errors, latents = compute_per_sequence_errors_and_latents(model, loader, torch.device("cpu"))
    assert errors.shape == (10,)
    assert np.allclose(errors, 4.0)
    assert latents.shape == (10, 4)


def test_compute_per_sequence_errors_empty_loader():
    loader = _make_loader(n=0)
    model = _ConstantAutoencoder(bias=1.0)
    errors, latents = compute_per_sequence_errors_and_latents(model, loader, torch.device("cpu"))
    assert len(errors) == 0


# ---------------------------------------------------------------------
# run_adwin_detector
# ---------------------------------------------------------------------


def test_adwin_detects_an_injected_mean_shift():
    rng = np.random.default_rng(0)
    stable = rng.normal(0.1, 0.02, 300)
    shifted = rng.normal(2.0, 0.02, 300)
    stream = np.concatenate([stable, shifted])

    drift_points = run_adwin_detector(stream, delta=0.002)
    assert len(drift_points) > 0
    # detected somewhere after the true shift (index 300), not wildly early
    assert all(idx >= 295 for idx in drift_points)


def test_adwin_no_drift_on_stationary_stream():
    rng = np.random.default_rng(1)
    stream = rng.normal(0.1, 0.02, 500)
    drift_points = run_adwin_detector(stream, delta=0.002)
    assert drift_points == []


# ---------------------------------------------------------------------
# compute_psi / compute_latent_psi
# ---------------------------------------------------------------------


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(0, 1, 2000)
    psi = compute_psi(reference, current, num_bins=10)
    assert abs(psi) < 0.05


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(5, 1, 2000)  # far-shifted
    psi = compute_psi(reference, current, num_bins=10)
    assert psi > 0.25  # well above the frozen spec's threshold


def test_psi_handles_near_constant_reference_gracefully():
    reference = np.zeros(100)
    current = np.ones(100)
    psi = compute_psi(reference, current, num_bins=10)
    assert psi == 0.0  # can't form bins from a constant reference -- documented no-op, not a crash


def test_compute_latent_psi_is_mean_across_dimensions():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, (2000, 2))
    # dim 0 shifted a lot, dim 1 unchanged
    current = np.stack([rng.normal(5, 1, 2000), rng.normal(0, 1, 2000)], axis=1)

    psi_dim0 = compute_psi(reference[:, 0], current[:, 0])
    psi_dim1 = compute_psi(reference[:, 1], current[:, 1])
    mean_psi = compute_latent_psi(reference, current)
    assert np.isclose(mean_psi, (psi_dim0 + psi_dim1) / 2, atol=1e-6)


# ---------------------------------------------------------------------
# evaluate_drift_stream
# ---------------------------------------------------------------------


def test_evaluate_drift_stream_triggers_retrain_on_persistent_drift():
    rng = np.random.default_rng(5)
    round_size = 50
    num_stable_rounds = 4
    num_drift_rounds = 4

    stable_errors = rng.normal(0.1, 0.01, round_size * num_stable_rounds)
    drift_errors = rng.normal(3.0, 0.01, round_size * num_drift_rounds)
    errors = np.concatenate([stable_errors, drift_errors])

    latent_dim = 3
    reference_latents = rng.normal(0, 1, (5000, latent_dim))
    stable_latents = rng.normal(0, 1, (round_size * num_stable_rounds, latent_dim))
    drift_latents = rng.normal(4, 1, (round_size * num_drift_rounds, latent_dim))
    latents = np.concatenate([stable_latents, drift_latents])

    result = evaluate_drift_stream(
        errors=errors, latents=latents, reference_latents=reference_latents,
        round_window_size=round_size, adwin_delta=0.002, psi_threshold=0.25,
        consecutive_rounds_for_retrain=2,
    )

    assert result["num_rounds"] == num_stable_rounds + num_drift_rounds
    assert result["first_trigger_round"] is not None
    # trigger must fire during/after the drifted rounds, never during the stable prefix
    assert result["first_trigger_round"] >= num_stable_rounds

    stable_reports = result["round_reports"][:num_stable_rounds]
    assert all(not r["retrain_triggered"] for r in stable_reports)


def test_evaluate_drift_stream_no_trigger_on_fully_stable_stream():
    # round_size=500: PSI needs enough samples/bin for a stable estimate --
    # verified empirically that round_size=50 (10 bins, ~5 samples/bin) lets
    # PSI spuriously exceed 0.25 on pure noise alone (one draw hit 0.461),
    # while >=200 stays comfortably below threshold under the null. This is
    # a real constraint on round_window_size for the actual data pipeline,
    # not just a test-tuning detail.
    rng = np.random.default_rng(6)
    round_size = 500
    num_rounds = 6
    errors = rng.normal(0.1, 0.01, round_size * num_rounds)
    latent_dim = 3
    reference_latents = rng.normal(0, 1, (5000, latent_dim))
    latents = rng.normal(0, 1, (round_size * num_rounds, latent_dim))

    result = evaluate_drift_stream(
        errors=errors, latents=latents, reference_latents=reference_latents,
        round_window_size=round_size, consecutive_rounds_for_retrain=2,
    )
    assert result["first_trigger_round"] is None
    assert all(not r["retrain_triggered"] for r in result["round_reports"])
