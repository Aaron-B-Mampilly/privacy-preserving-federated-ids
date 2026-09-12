"""Phase 11: loss-based membership inference attack (MIA), the standard
way to empirically validate whether Phase 8's client-level DP mechanism
actually reduces privacy leakage -- not just theoretically, but as
measured by a real attack.

Mechanism (classic loss-threshold MIA, Yeom et al. 2018 / Shokri et al.,
no shadow models needed): a model trained on some examples tends to
have LOWER loss on those examples ("members") than on examples it never
saw ("non-members"). An attacker exploiting only this gap, with no
other information, is modeled as a threshold classifier on per-example
loss; its ROC AUC and "membership advantage" (max TPR-FPR over the ROC
curve, the standard privacy-leakage metric from Yeom et al.) measure
how much the model's loss alone reveals about training-set membership.

Per-client formulation (matches this project's personalized FL, where
each client has its own classifier head): "member" = that client's own
TRAIN split (used to train both the shared encoder/decoder and this
client's own head); "non-member" = that SAME client's own TEST split
(same distribution/client, structurally held out, never trained on) --
the standard train-vs-test MIA setup, requiring no cross-client head
swapping.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from fedpda_ids.models.lstm_autoencoder import compute_total_loss


@torch.no_grad()
def compute_per_example_losses(model: torch.nn.Module, loader: DataLoader, device: torch.device, lambda_ce: float) -> np.ndarray:
    """Per-example total_loss (MSE_recon + lambda_ce * CE) -- the exact
    objective the model was trained on, never batch-averaged (the MIA
    needs one loss value per example, not one per batch)."""
    model.eval()
    all_losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        reconstruction, logits, _latent = model(x)
        mse_per_example = torch.mean((reconstruction - x) ** 2, dim=tuple(range(1, x.dim())))
        ce_per_example = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        all_losses.append((mse_per_example + lambda_ce * ce_per_example).cpu().numpy())
    if not all_losses:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(all_losses)


def run_loss_threshold_mia(member_losses: np.ndarray, non_member_losses: np.ndarray) -> dict:
    """Classic loss-threshold MIA: score = -loss (lower loss -> more
    likely member, so negate for a monotonic "membership probability"
    usable directly with sklearn's AUC/ROC utilities). Returns AUC,
    membership advantage (max TPR-FPR over the ROC curve, Yeom et al.'s
    definition), and the mean loss gap (member should be lower)."""
    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return {
            "auc": float("nan"), "advantage": float("nan"),
            "mean_member_loss": float("nan"), "mean_non_member_loss": float("nan"),
            "num_member": len(member_losses), "num_non_member": len(non_member_losses),
        }

    y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    scores = np.concatenate([-member_losses, -non_member_losses])

    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    advantage = float(np.max(tpr - fpr))

    return {
        "auc": auc,
        "advantage": advantage,
        "mean_member_loss": float(np.mean(member_losses)),
        "mean_non_member_loss": float(np.mean(non_member_losses)),
        "num_member": int(len(member_losses)),
        "num_non_member": int(len(non_member_losses)),
    }
