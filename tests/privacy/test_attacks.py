"""Phase 11: loss-threshold membership inference attack (MIA) tests."""

import numpy as np
import torch

from fedpda_ids.privacy.attacks import compute_per_example_losses, run_loss_threshold_mia


# ---------------------------------------------------------------------
# compute_per_example_losses
# ---------------------------------------------------------------------


class _KnownLossModel(torch.nn.Module):
    """Reconstructs input exactly (mse=0) and always predicts class 0
    with a fixed logit gap -- lets tests control per-example CE loss
    precisely via the label."""

    def forward(self, x):
        reconstruction = x.clone()
        logits = torch.zeros(x.shape[0], 2)
        logits[:, 0] = 10.0  # strongly favors class 0
        latent = x[:, -1, :]
        return reconstruction, logits, latent


def _make_loader(labels, window=5, num_features=2, batch_size=4):
    n = len(labels)
    x = torch.zeros(n, window, num_features)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_compute_per_example_losses_is_mse_plus_lambda_ce():
    # class 0 (matches the model's strong prediction) -> near-zero CE;
    # class 1 (mismatched) -> large CE. Reconstruction is exact (mse=0) always.
    loader = _make_loader(labels=[0, 1, 0, 1])
    model = _KnownLossModel()
    losses = compute_per_example_losses(model, loader, torch.device("cpu"), lambda_ce=1.0)
    assert losses.shape == (4,)
    assert losses[0] < 0.01  # class 0: mse=0, ce~=0
    assert losses[1] > 5.0   # class 1: mse=0, ce large (mismatched prediction)
    assert np.isclose(losses[0], losses[2])
    assert np.isclose(losses[1], losses[3])


def test_compute_per_example_losses_empty_loader():
    loader = _make_loader(labels=[])
    model = _KnownLossModel()
    losses = compute_per_example_losses(model, loader, torch.device("cpu"), lambda_ce=1.0)
    assert len(losses) == 0


# ---------------------------------------------------------------------
# run_loss_threshold_mia
# ---------------------------------------------------------------------


def test_mia_high_auc_when_members_have_systematically_lower_loss():
    rng = np.random.default_rng(0)
    member_losses = rng.normal(0.1, 0.02, 500)      # members: low loss (model fit them well)
    non_member_losses = rng.normal(1.0, 0.02, 500)  # non-members: high loss

    result = run_loss_threshold_mia(member_losses, non_member_losses)
    assert result["auc"] > 0.95
    assert result["advantage"] > 0.8
    assert result["mean_member_loss"] < result["mean_non_member_loss"]


def test_mia_auc_near_half_when_no_real_gap():
    rng = np.random.default_rng(1)
    member_losses = rng.normal(0.5, 0.1, 1000)
    non_member_losses = rng.normal(0.5, 0.1, 1000)  # same distribution -- no real membership signal

    result = run_loss_threshold_mia(member_losses, non_member_losses)
    assert 0.4 < result["auc"] < 0.6
    assert result["advantage"] < 0.15


def test_mia_handles_empty_inputs_without_crashing():
    result = run_loss_threshold_mia(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(result["auc"])
    assert result["num_member"] == 0
    assert result["num_non_member"] == 2


def test_mia_partial_gap_gives_intermediate_auc():
    # a smaller, more realistic gap should give an AUC clearly between
    # the "no gap" and "huge gap" cases, not saturated at 0.5 or 1.0.
    rng = np.random.default_rng(2)
    member_losses = rng.normal(0.45, 0.15, 800)
    non_member_losses = rng.normal(0.55, 0.15, 800)

    result = run_loss_threshold_mia(member_losses, non_member_losses)
    assert 0.5 < result["auc"] < 0.95
