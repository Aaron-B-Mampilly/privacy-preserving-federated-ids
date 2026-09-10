"""Phase 4 Part J: DNN ablation baseline unit tests."""

import torch

from fedpda_ids.models.dnn_baseline import DNNBaseline

F = 70
C = 5
W = 10
B = 8


def test_dnn_output_shape():
    model = DNNBaseline(num_features=F, num_classes=C)
    x = torch.randn(B, W, F)
    logits = model(x)
    assert logits.shape == (B, C)


def test_dnn_only_uses_last_timestep():
    """Changing everything EXCEPT the last timestep must not change the
    output -- this is what makes it a genuinely non-temporal baseline."""
    model = DNNBaseline(num_features=F, num_classes=C)
    model.eval()
    x1 = torch.randn(B, W, F)
    x2 = x1.clone()
    x2[:, :-1, :] = torch.randn(B, W - 1, F)  # scramble everything but the last row

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert torch.allclose(out1, out2)


def test_dnn_backward_and_optimizer_step():
    model = DNNBaseline(num_features=F, num_classes=C)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.clone().detach() for p in model.parameters()]

    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_dnn_no_nan_or_inf():
    model = DNNBaseline(num_features=F, num_classes=C)
    x = torch.randn(B, W, F)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_dnn_comparable_parameter_scale_to_lstm_encoder_and_head():
    from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier

    dnn = DNNBaseline(num_features=F, num_classes=C)
    lstm = LSTMAutoencoderClassifier(num_features=F, num_classes=C, window_size=W)

    dnn_params = sum(p.numel() for p in dnn.parameters())
    lstm_encoder_and_head_params = (
        sum(p.numel() for p in lstm.encoder.parameters())
        + sum(p.numel() for p in lstm.classifier.parameters())
    )
    # not equal (different layer types), but same order of magnitude
    ratio = dnn_params / lstm_encoder_and_head_params
    assert 0.1 < ratio < 10
