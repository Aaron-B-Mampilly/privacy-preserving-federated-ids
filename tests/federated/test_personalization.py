"""Phase 6 Part C: parameter-ownership unit tests (Tests 1-7 from the
Phase 6 spec; Tests 8-11 live in test_personalized_simulation.py since
they need the full simulation harness).
"""

import numpy as np
import torch

from fedpda_ids.federated.client import (
    FlowerLSTMClient,
    get_shared_parameter_keys,
    get_shared_parameters,
    load_local_head,
    save_local_head,
    set_shared_parameters,
)
from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier

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


def test_shared_parameter_keys_are_exactly_encoder_and_decoder():
    model = _make_model()
    keys = get_shared_parameter_keys(model)
    assert len(keys) == 16  # 8 encoder + 8 decoder
    assert all(k.startswith("encoder.") or k.startswith("decoder.") for k in keys)
    assert not any(k.startswith("classifier.") for k in keys)


# TEST 1: shared encoder parameters are federated (i.e. round-trip through get/set_shared)
def test_encoder_parameters_are_federated():
    model_a = _make_model()
    model_b = _make_model()
    shared_a = get_shared_parameters(model_a)
    set_shared_parameters(model_b, shared_a)

    keys_a = model_a.state_dict()
    keys_b = model_b.state_dict()
    for k in keys_a:
        if k.startswith("encoder."):
            assert np.array_equal(keys_a[k].numpy(), keys_b[k].numpy())


# TEST 2: shared decoder parameters are federated
def test_decoder_parameters_are_federated():
    model_a = _make_model()
    model_b = _make_model()
    shared_a = get_shared_parameters(model_a)
    set_shared_parameters(model_b, shared_a)

    keys_a = model_a.state_dict()
    keys_b = model_b.state_dict()
    for k in keys_a:
        if k.startswith("decoder."):
            assert np.array_equal(keys_a[k].numpy(), keys_b[k].numpy())


# TEST 3: local classification head is NOT federated
def test_classifier_head_is_not_federated():
    model_a = _make_model()
    model_b = _make_model()
    classifier_b_before = {k: v.clone() for k, v in model_b.state_dict().items() if k.startswith("classifier.")}

    shared_a = get_shared_parameters(model_a)
    set_shared_parameters(model_b, shared_a)

    classifier_b_after = {k: v for k, v in model_b.state_dict().items() if k.startswith("classifier.")}
    for k in classifier_b_before:
        assert torch.equal(classifier_b_before[k], classifier_b_after[k])


# TEST 4: client A's head remains unchanged by client B's training
def test_client_a_head_unchanged_by_client_b_training(tmp_path):
    model_a = _make_model()
    model_b = _make_model()
    head_a_path = tmp_path / "heads" / "client_0.pt"
    head_b_path = tmp_path / "heads" / "client_1.pt"

    client_a = FlowerLSTMClient(
        client_id=0, model=model_a, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=1, lambda_ce=1.0, learning_rate=1e-2,
        index_to_label={i: f"c{i}" for i in range(C)}, personalized=True, head_path=head_a_path,
    )
    client_b = FlowerLSTMClient(
        client_id=1, model=model_b, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=1, lambda_ce=1.0, learning_rate=1e-2,
        index_to_label={i: f"c{i}" for i in range(C)}, personalized=True, head_path=head_b_path,
    )

    shared_params = client_a.get_parameters({})
    client_a.fit(shared_params, {})
    head_a_after_a_fit = torch.load(head_a_path, weights_only=True)

    # now train client B -- must not touch client A's saved head file at all
    client_b.fit(shared_params, {})

    head_a_after_b_fit = torch.load(head_a_path, weights_only=True)
    for k in head_a_after_a_fit:
        assert torch.equal(head_a_after_a_fit[k], head_a_after_b_fit[k])


# TEST 5: different clients can maintain different head parameters
def test_different_clients_maintain_different_heads(tmp_path):
    model_a = _make_model()
    model_b = _make_model()
    head_a_path = tmp_path / "heads" / "client_0.pt"
    head_b_path = tmp_path / "heads" / "client_1.pt"

    client_a = FlowerLSTMClient(
        client_id=0, model=model_a, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=2, lambda_ce=1.0, learning_rate=1e-2,
        index_to_label={i: f"c{i}" for i in range(C)}, personalized=True, head_path=head_a_path,
    )
    client_b = FlowerLSTMClient(
        client_id=1, model=model_b, train_loader=_make_loader(), val_loader=_make_loader(),
        device=torch.device("cpu"), local_epochs=2, lambda_ce=1.0, learning_rate=1e-2,
        index_to_label={i: f"c{i}" for i in range(C)}, personalized=True, head_path=head_b_path,
    )

    shared_params = client_a.get_parameters({})
    client_a.fit(shared_params, {})
    client_b.fit(shared_params, {})

    head_a = torch.load(head_a_path, weights_only=True)
    head_b = torch.load(head_b_path, weights_only=True)
    assert any(not torch.equal(head_a[k], head_b[k]) for k in head_a)


# TEST 6: aggregation changes only shared parameters (server-side simulation
# of FedAvg averaging -- two clients' shared params average correctly,
# independent of their (different) local heads)
def test_fedavg_averaging_only_touches_shared_params():
    model_a = _make_model()
    model_b = _make_model()
    shared_a = get_shared_parameters(model_a)
    shared_b = get_shared_parameters(model_b)

    averaged = [(a + b) / 2.0 for a, b in zip(shared_a, shared_b)]

    merged_model = _make_model()
    classifier_before = {k: v.clone() for k, v in merged_model.state_dict().items() if k.startswith("classifier.")}
    set_shared_parameters(merged_model, averaged)
    classifier_after = {k: v for k, v in merged_model.state_dict().items() if k.startswith("classifier.")}

    for k in classifier_before:
        assert torch.equal(classifier_before[k], classifier_after[k])

    merged_shared = get_shared_parameters(merged_model)
    for m, a in zip(merged_shared, averaged):
        assert np.allclose(m, a)


# TEST 7: shared parameter shapes remain correct
def test_shared_parameter_shapes_correct():
    model = _make_model()
    shared = get_shared_parameters(model)
    keys = get_shared_parameter_keys(model)
    state_dict = model.state_dict()
    for k, p in zip(keys, shared):
        assert p.shape == tuple(state_dict[k].shape)


def test_load_local_head_returns_false_when_never_saved(tmp_path):
    model = _make_model()
    found = load_local_head(model, tmp_path / "heads" / "client_99.pt")
    assert found is False


def test_load_local_head_returns_true_and_restores_values(tmp_path):
    model_a = _make_model()
    path = tmp_path / "heads" / "client_0.pt"
    save_local_head(model_a, path)

    model_b = _make_model()  # different random init
    found = load_local_head(model_b, path)
    assert found is True

    for k in model_a.state_dict():
        if k.startswith("classifier."):
            assert torch.equal(model_a.state_dict()[k], model_b.state_dict()[k])
