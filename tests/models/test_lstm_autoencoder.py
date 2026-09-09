"""Phase 4 Part C: model architecture unit tests (Tests 1-6, 8-9)."""

import torch

from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier, compute_total_loss

F = 70
C = 5
W = 10
B = 8


def _make_model():
    return LSTMAutoencoderClassifier(num_features=F, num_classes=C, window_size=W)


def test_encoder_output_shape(): # TEST 1
    model = _make_model()
    x = torch.randn(B, W, F)
    latent = model.encoder(x)
    assert latent.shape == (B, 32)


def test_decoder_output_shape(): # TEST 2
    model = _make_model()
    latent = torch.randn(B, 32)
    recon = model.decoder(latent)
    assert recon.shape == (B, W, F)


def test_classifier_output_shape(): # TEST 3
    model = _make_model()
    latent = torch.randn(B, 32)
    logits = model.classifier(latent)
    assert logits.shape == (B, C)


def test_total_loss_is_finite(): # TEST 4
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    recon, logits, _ = model(x)
    total, mse, ce = compute_total_loss(recon, x, logits, y, lambda_ce=1.0)
    assert torch.isfinite(total)
    assert torch.isfinite(mse)
    assert torch.isfinite(ce)


def test_backward_pass_succeeds(): # TEST 5
    model = _make_model()
    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    recon, logits, _ = model(x)
    total, _, _ = compute_total_loss(recon, x, logits, y)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)


def test_optimizer_step_changes_parameters(): # TEST 6
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.clone().detach() for p in model.parameters()]

    x = torch.randn(B, W, F)
    y = torch.randint(0, C, (B,))
    recon, logits, _ = model(x)
    total, _, _ = compute_total_loss(recon, x, logits, y)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    after = list(model.parameters())
    changed = [not torch.equal(b, a) for b, a in zip(before, after)]
    assert any(changed)


def test_no_nan_or_inf_in_model_output(): # TEST 8
    model = _make_model()
    x = torch.randn(B, W, F)
    recon, logits, latent = model(x)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(latent).all()


def test_incomplete_final_batch_shapes_are_correct(): # TEST 9
    model = _make_model()
    x = torch.randn(3, W, F)  # smaller than a normal batch
    y = torch.randint(0, C, (3,))
    recon, logits, latent = model(x)
    assert latent.shape == (3, 32)
    assert recon.shape == (3, W, F)
    assert logits.shape == (3, C)
    total, _, _ = compute_total_loss(recon, x, logits, y)
    assert torch.isfinite(total)


def test_single_sample_batch_works(): # extra: batch size 1 (extreme non-IID client case)
    model = _make_model()
    x = torch.randn(1, W, F)
    y = torch.randint(0, C, (1,))
    recon, logits, latent = model(x)
    assert latent.shape == (1, 32)
    assert recon.shape == (1, W, F)
    assert logits.shape == (1, C)
