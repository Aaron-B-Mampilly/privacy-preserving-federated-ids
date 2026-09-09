"""Phase 4 Part C/E: checkpoint reproducibility (Test 10) and the full
synthetic end-to-end smoke test (model + Dataset/DataLoader + trainer),
run BEFORE any real-data training.
"""

import numpy as np
import pandas as pd
import torch

from fedpda_ids.data.sequence_dataset import build_scope_dataloaders
from fedpda_ids.data.sequences import build_sequences
from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier
from fedpda_ids.models.trainer import (
    load_checkpoint,
    save_checkpoint,
    select_device,
    train_model,
)

F = 6
C = 3
W = 10


def _rows(host, split, client, values, labels):
    n = len(values)
    data = {"host": [host] * n, "temporal_split": [split] * n, "client_id": [client] * n, "time": values}
    for f in range(F):
        data[f"f{f}"] = [v + f for v in values]
    data["Label"] = labels
    return pd.DataFrame(data)


def _build_synthetic_seq_dir(tmp_path):
    rng_labels = (["BENIGN"] * 6 + ["ATTACK"] * 6 + ["RARE"] * 6) * 2  # 36 values, 3 classes
    frames = [
        _rows("H", "train", 0, list(range(0, 60)), (rng_labels * 2)[:60]),
        _rows("H", "val", 0, list(range(100, 130)), (rng_labels)[:30]),
        _rows("H", "test", 0, list(range(200, 230)), (rng_labels)[:30]),
    ]
    df = pd.concat(frames, ignore_index=True)
    output_dir = tmp_path / "seqs"
    build_sequences(
        df,
        feature_cols=[f"f{f}" for f in range(F)],
        base_group_cols=["host"],
        time_col="time",
        label_col="Label",
        client_id_col="client_id",
        window_size=W,
        stride=5,
        output_dir=output_dir,
    )
    return output_dir


def test_checkpoint_save_load_reproduces_identical_inference(tmp_path):  # TEST 10
    torch.manual_seed(0)
    model = LSTMAutoencoderClassifier(num_features=F, num_classes=C, window_size=W)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, W, F)
    model.eval()
    with torch.no_grad():
        recon_before, logits_before, _ = model(x)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch=1, run_config={"seed": 0}, train_metrics={}, val_metrics={})

    # fresh model instance, then load
    model2 = LSTMAutoencoderClassifier(num_features=F, num_classes=C, window_size=W)
    ckpt = load_checkpoint(ckpt_path)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()

    with torch.no_grad():
        recon_after, logits_after, _ = model2(x)

    assert torch.allclose(recon_before, recon_after)
    assert torch.allclose(logits_before, logits_after)
    assert ckpt["epoch"] == 1
    assert ckpt["run_config"] == {"seed": 0}


def test_device_selection_prints_and_returns_cpu_without_cuda(capsys):
    device = select_device()
    assert device.type in ("cpu", "cuda")
    captured = capsys.readouterr()
    assert "torch version" in captured.out
    assert "CUDA available" in captured.out
    assert "device" in captured.out


def test_synthetic_end_to_end_smoke_test(tmp_path):
    """PART E: full pipeline on tiny synthetic data -- forward, finite
    losses, backward, checkpoint save/load, metrics compute."""
    seq_dir = _build_synthetic_seq_dir(tmp_path)
    scope = build_scope_dataloaders(seq_dir, client_id_col="client_id", client_id=0, batch_size=4)

    assert len(scope["datasets"]["train"]) > 0
    assert len(scope["datasets"]["val"]) > 0

    device = select_device()
    model = LSTMAutoencoderClassifier(num_features=F, num_classes=len(scope["label_to_index"]), window_size=W).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = train_model(
        model,
        scope["loaders"]["train"],
        scope["loaders"]["val"],
        optimizer,
        device,
        epochs=2,
        lambda_ce=1.0,
        index_to_label=scope["index_to_label"],
        checkpoint_dir=tmp_path / "checkpoints",
        run_name="smoke_test",
        run_config={"dataset": "synthetic", "client_id": 0, "num_features": F},
    )

    assert result["best_epoch"] in (1, 2)
    assert np.isfinite(result["best_val_loss"])
    for epoch_record in result["history"]["train"] + result["history"]["val"]:
        assert np.isfinite(epoch_record["loss"])

    best_ckpt = load_checkpoint(result["best_checkpoint"])
    assert "model_state_dict" in best_ckpt
    assert best_ckpt["run_config"]["dataset"] == "synthetic"

    last_ckpt_path = tmp_path / "checkpoints" / "smoke_test_last.pt"
    assert last_ckpt_path.exists()
