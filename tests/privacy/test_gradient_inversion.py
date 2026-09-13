"""E6's gradient inversion attack (DLG). Uses a deliberately tiny model
(latent_dim=8, W=5, F=4) so these tests stay fast -- the mechanism
(double-backward through the SAME LSTM architecture) is what's under
test, not full-scale reconstruction quality.
"""

import numpy as np
import torch

from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier
from fedpda_ids.privacy.dp import add_gaussian_noise, clip_update
from fedpda_ids.privacy.gradient_inversion import (
    aggregate_gradients,
    compute_client_gradient,
    dlg_reconstruct,
    reconstruction_mse,
)

F = 4
C = 2
W = 5
B = 1


def _make_model():
    return LSTMAutoencoderClassifier(
        num_features=F, num_classes=C, window_size=W, latent_dim=8,
        encoder_layer1_units=8, decoder_layer1_units=8, classifier_hidden_units=8,
    )


def test_compute_client_gradient_matches_param_count():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))

    grad = compute_client_gradient(model, x, y, lambda_ce=1.0)

    assert len(grad) == len(list(model.parameters()))
    assert all(torch.isfinite(g).all() for g in grad)


def test_compute_client_gradient_shared_only_restricts_to_encoder_decoder():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))

    shared_grad = compute_client_gradient(model, x, y, lambda_ce=1.0, shared_only_prefixes=("encoder.", "decoder."))
    num_shared_params = sum(1 for name, _ in model.named_parameters() if name.startswith(("encoder.", "decoder.")))

    assert len(shared_grad) == num_shared_params
    assert num_shared_params < len(list(model.parameters()))  # sanity: classifier params really were excluded


# TEST: DLG's grad-matching objective actually decreases under optimization
# (the reconstruction is doing real work, not returning noise unchanged).
def test_dlg_reconstruct_reduces_grad_matching_distance():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    target_grad = compute_client_gradient(model, x, y, lambda_ce=1.0)

    result = dlg_reconstruct(model, target_grad, x.shape, C, lambda_ce=1.0, num_iterations=40, lr=0.5, seed=1)

    history = result["grad_distance_history"]
    assert len(history) == 40
    assert history[-1] < history[0]
    assert result["final_grad_distance"] == history[-1]


# TEST: given ENOUGH iterations on an unprotected (raw) gradient, DLG
# recovers the real input to near-exact precision -- this is DLG's core
# published result (Zhu et al. 2019), and the whole reason gradient
# inversion is treated as a real privacy attack rather than a curiosity.
def test_dlg_reconstruct_recovers_unprotected_gradient_near_exactly():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    target_grad = compute_client_gradient(model, x, y, lambda_ce=1.0)

    result = dlg_reconstruct(model, target_grad, x.shape, C, lambda_ce=1.0, num_iterations=100, lr=0.5, seed=1)
    mse = reconstruction_mse(x, result["dummy_x"])

    assert mse < 0.01  # real x has unit variance -- this is near-exact recovery


# TEST (E6's "Ours+DP" comparator): clipping+noising the SAME gradient
# before it's attacked makes reconstruction dramatically WORSE than the
# unprotected case -- DP is actually doing its job against this attack.
def test_dp_clipped_noised_gradient_reconstruction_worse_than_raw():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    target_grad = compute_client_gradient(model, x, y, lambda_ce=1.0)

    raw_result = dlg_reconstruct(model, target_grad, x.shape, C, lambda_ce=1.0, num_iterations=100, lr=0.5, seed=1)
    raw_mse = reconstruction_mse(x, raw_result["dummy_x"])

    grad_np = [g.numpy() for g in target_grad]
    clipped = clip_update(grad_np, clip_norm=0.01)
    noised = add_gaussian_noise(clipped, noise_multiplier=5.0, clip_norm=0.01, rng=np.random.default_rng(0))
    noised_grad = [torch.tensor(a) for a in noised]

    dp_result = dlg_reconstruct(model, noised_grad, x.shape, C, lambda_ce=1.0, num_iterations=100, lr=0.5, seed=1)
    dp_mse = reconstruction_mse(x, dp_result["dummy_x"])

    assert dp_mse > raw_mse * 10  # generous margin -- the point is "much worse," not a precise ratio


# TEST (E6's "Ours+DP+SecAgg" comparator): the attacker only ever sees
# the MEAN across clients (never one client's own contribution) under
# SecAgg+ -- attacking that mean while targeting one specific client's
# real input is harder than attacking that client's own raw gradient
# directly (the real privacy benefit SecAgg+ adds beyond DP alone).
def test_averaged_gradient_reconstruction_worse_than_targeted_clients_own_raw_gradient():
    torch.manual_seed(0)
    model = _make_model()
    x_target = torch.randn(B, W, F)
    y_target = torch.randint(0, C, (B,))
    x_other = torch.randn(B, W, F) + 5.0  # a clearly different other client's real input
    y_other = torch.randint(0, C, (B,))

    target_grad = compute_client_gradient(model, x_target, y_target, lambda_ce=1.0)
    other_grad = compute_client_gradient(model, x_other, y_other, lambda_ce=1.0)

    raw_result = dlg_reconstruct(model, target_grad, x_target.shape, C, lambda_ce=1.0, num_iterations=100, lr=0.5, seed=1)
    raw_mse = reconstruction_mse(x_target, raw_result["dummy_x"])

    averaged_grad = aggregate_gradients([target_grad, other_grad])
    agg_result = dlg_reconstruct(model, averaged_grad, x_target.shape, C, lambda_ce=1.0, num_iterations=100, lr=0.5, seed=1)
    agg_mse = reconstruction_mse(x_target, agg_result["dummy_x"])

    assert agg_mse > raw_mse


def test_aggregate_gradients_is_elementwise_mean():
    a = [torch.tensor([2.0, 4.0]), torch.tensor([[1.0, 1.0]])]
    b = [torch.tensor([0.0, 0.0]), torch.tensor([[3.0, 5.0]])]

    aggregated = aggregate_gradients([a, b])

    assert torch.allclose(aggregated[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(aggregated[1], torch.tensor([[2.0, 3.0]]))
