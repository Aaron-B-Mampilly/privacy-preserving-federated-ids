"""Phase 4: training loop, evaluation, checkpointing, device selection.

Best-checkpoint criterion (documented, not implicit): the checkpoint
saved as "<run_name>_best.pt" is the epoch with the LOWEST validation
total loss (MSE + lambda*CE) -- the same objective being optimized,
computed on held-out data. "<run_name>_last.pt" is always the final
epoch's checkpoint. Only these two files are ever written per run, to
avoid uncontrolled duplicate checkpoints piling up every epoch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedpda_ids.evaluation.metrics import compute_classification_metrics
from fedpda_ids.models.lstm_autoencoder import compute_total_loss

logger = logging.getLogger("fedpda_ids")


def select_device() -> torch.device:
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"device: {device}")
    return device


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_ce: float,
) -> dict:
    model.train()
    loss_sum = mse_sum = ce_sum = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        reconstruction, logits, _ = model(x)
        total, mse, ce = compute_total_loss(reconstruction, x, logits, y, lambda_ce)
        total.backward()
        optimizer.step()

        loss_sum += total.item()
        mse_sum += mse.item()
        ce_sum += ce.item()
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {"loss": loss_sum / n_batches, "mse": mse_sum / n_batches, "ce": ce_sum / n_batches}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_ce: float,
    index_to_label: dict[int, str],
) -> dict:
    model.eval()
    loss_sum = mse_sum = ce_sum = 0.0
    n_batches = 0
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        reconstruction, logits, _ = model(x)
        total, mse, ce = compute_total_loss(reconstruction, x, logits, y, lambda_ce)

        loss_sum += total.item()
        mse_sum += mse.item()
        ce_sum += ce.item()
        n_batches += 1

        all_y_true.append(y.cpu().numpy())
        all_y_pred.append(logits.argmax(dim=1).cpu().numpy())

    n_batches = max(n_batches, 1)
    y_true = np.concatenate(all_y_true) if all_y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_y_pred) if all_y_pred else np.array([], dtype=np.int64)

    metrics = compute_classification_metrics(y_true, y_pred, index_to_label)
    metrics["loss"] = loss_sum / n_batches
    metrics["mse"] = mse_sum / n_batches
    metrics["ce"] = ce_sum / n_batches
    return metrics


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_config: dict[str, Any],
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "run_config": run_config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    # weights_only=False (explicit, not relying on torch-version defaults):
    # this checkpoint carries run_config/metrics dicts, not just tensors.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    lambda_ce: float,
    index_to_label: dict[int, str],
    checkpoint_dir: str | Path,
    run_name: str,
    run_config: dict[str, Any],
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    best_val_loss = float("inf")
    best_epoch = -1
    history: dict[str, list[dict]] = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, lambda_ce)
        val_metrics = evaluate(model, val_loader, device, lambda_ce, index_to_label)

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        logger.info(
            "[%s] epoch %d/%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f val_acc=%.4f",
            run_name, epoch, epochs, train_metrics["loss"], val_metrics["loss"],
            val_metrics["macro_f1"], val_metrics["accuracy"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_dir / f"{run_name}_best.pt", model, optimizer, epoch,
                run_config, train_metrics, val_metrics,
            )

    save_checkpoint(
        checkpoint_dir / f"{run_name}_last.pt", model, optimizer, epochs,
        run_config, history["train"][-1], history["val"][-1],
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoint_dir / f"{run_name}_best.pt"),
        "last_checkpoint": str(checkpoint_dir / f"{run_name}_last.pt"),
    }
