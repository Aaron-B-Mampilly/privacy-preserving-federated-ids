"""Phase 5: FlowerLSTMClient unit tests (no Ray/simulation harness --
direct method calls, fast and isolated from Flower's process model)."""

import numpy as np
import pytest
import torch

from fedpda_ids.federated.client import FlowerLSTMClient, get_model_parameters, get_shared_parameters, set_model_parameters
from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier
from fedpda_ids.privacy.dp import update_norm

F = 10
C = 3
W = 10
B = 8


def _make_model():
    return LSTMAutoencoderClassifier(num_features=F, num_classes=C, window_size=W)


def _make_loader(n=16):
    x = torch.randn(n, W, F)
    y = torch.randint(0, C, (n,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def test_get_set_parameters_roundtrip():
    model_a = _make_model()
    model_b = _make_model()  # different random init
    params_a = get_model_parameters(model_a)
    set_model_parameters(model_b, params_a)
    params_b = get_model_parameters(model_b)
    for pa, pb in zip(params_a, params_b):
        assert np.array_equal(pa, pb)


def test_get_parameters_matches_state_dict_order():
    model = _make_model()
    params = get_model_parameters(model)
    state_dict_values = list(model.state_dict().values())
    assert len(params) == len(state_dict_values)
    for p, v in zip(params, state_dict_values):
        assert np.array_equal(p, v.numpy())


def test_client_fit_returns_correct_types_and_trains():
    model = _make_model()
    initial_params = get_model_parameters(model)

    client = FlowerLSTMClient(
        client_id=0, model=model, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=2, lambda_ce=1.0, learning_rate=1e-3,
        index_to_label={i: f"class{i}" for i in range(C)},
    )

    new_params, num_examples, metrics = client.fit(initial_params, {})

    assert isinstance(new_params, list)
    assert all(isinstance(p, np.ndarray) for p in new_params)
    assert num_examples == 16
    assert "train_loss" in metrics
    assert np.isfinite(metrics["train_loss"])

    # weights should have actually changed after local training
    assert any(not np.array_equal(a, b) for a, b in zip(initial_params, new_params))


def test_client_evaluate_returns_correct_types():
    model = _make_model()
    params = get_model_parameters(model)

    client = FlowerLSTMClient(
        client_id=0, model=model, train_loader=_make_loader(), val_loader=_make_loader(n=12),
        device=torch.device("cpu"), local_epochs=1, lambda_ce=1.0, learning_rate=1e-3,
        index_to_label={i: f"class{i}" for i in range(C)},
    )

    loss, num_examples, metrics = client.evaluate(params, {})

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert num_examples == 12
    assert "accuracy" in metrics and "macro_f1" in metrics


def test_fit_does_not_mutate_input_parameters_list_identity():
    """The parameters Flower hands to fit() represent the GLOBAL model --
    fit() must not corrupt the caller's copy (Flower re-serializes the
    return value, but a client silently aliasing and mutating the input
    arrays in place would be a correctness trap for any future caller
    that keeps a reference)."""
    model = _make_model()
    params = get_model_parameters(model)
    params_copy = [p.copy() for p in params]

    client = FlowerLSTMClient(
        client_id=0, model=model, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=1, lambda_ce=1.0, learning_rate=1e-3,
        index_to_label={i: f"class{i}" for i in range(C)},
    )
    client.fit(params, {})

    for original, copy in zip(params, params_copy):
        assert np.array_equal(original, copy)


# TEST (E6's DP+SecAgg+ comparator): dp_clip_norm mode returns a CLIPPED
# DELTA -- not raw parameters -- with its L2 norm at most clip_norm, and
# forces num_examples=1 so SecAgg+'s per-client weighting is uniform.
def test_dp_clip_norm_returns_clipped_delta_within_bound_and_forces_num_examples_1(tmp_path):
    model = _make_model()
    initial_params = get_shared_parameters(model)
    clip_norm = 1e-4  # tiny on purpose -- guarantees the real update norm exceeds it, so clipping is exercised

    client = FlowerLSTMClient(
        client_id=0, model=model, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=2, lambda_ce=1.0, learning_rate=1e-2,
        index_to_label={i: f"class{i}" for i in range(C)},
        personalized=True, head_path=tmp_path / "head.pt",
        dp_clip_norm=clip_norm,
    )

    returned_delta, num_examples, _ = client.fit(initial_params, {})

    assert num_examples == 1
    assert len(returned_delta) == len(initial_params)
    assert update_norm(returned_delta) <= clip_norm + 1e-8


def test_dp_clip_norm_requires_personalized_true():
    model = _make_model()
    with pytest.raises(AssertionError):
        FlowerLSTMClient(
            client_id=0, model=model, train_loader=_make_loader(), val_loader=_make_loader(),
            device=torch.device("cpu"), local_epochs=1, lambda_ce=1.0, learning_rate=1e-3,
            index_to_label={i: f"class{i}" for i in range(C)},
            personalized=False, dp_clip_norm=1.0,
        )
