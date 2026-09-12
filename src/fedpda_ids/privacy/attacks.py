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
    """Loss-threshold MIA, checked in BOTH directions and reported as
    whichever is stronger -- a real attacker isn't obligated to
    precommit to "lower loss = member" (the textbook assumption); they
    would use whichever direction is empirically more predictive.

    This matters concretely for this project: DP-trained checkpoints
    (real result, found via this exact function) show member loss
    HIGHER than non-member loss on average -- the reverse of the
    classic assumption, plausibly from the shared encoder and a
    client's persisted head being trained against DIFFERENT rounds'
    (differently-noised) encoder states under heavy DP noise. Reporting
    only the "lower loss = member" direction would have silently
    UNDERSTATED the true attack risk for exactly those checkpoints --
    the case that matters most for validating whether DP protects
    against this attack. `attack_direction` in the returned dict names
    which convention won, so this is never hidden.

    Returns the WINNING direction's AUC/advantage as "auc"/"advantage"
    (this is what should be compared across the epsilon sweep), plus
    both directions' AUC for full transparency."""
    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return {
            "auc": float("nan"), "advantage": float("nan"), "attack_direction": None,
            "auc_lower_loss_is_member": float("nan"), "auc_higher_loss_is_member": float("nan"),
            "mean_member_loss": float("nan"), "mean_non_member_loss": float("nan"),
            "num_member": len(member_losses), "num_non_member": len(non_member_losses),
        }

    y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    losses = np.concatenate([member_losses, non_member_losses])

    # Direction A: lower loss -> member (the textbook assumption).
    auc_lower = float(roc_auc_score(y_true, -losses))
    fpr_lower, tpr_lower, _ = roc_curve(y_true, -losses)
    advantage_lower = float(np.max(tpr_lower - fpr_lower))

    # Direction B: higher loss -> member (what a rational attacker
    # would switch to if this direction is actually more predictive).
    auc_higher = float(roc_auc_score(y_true, losses))
    fpr_higher, tpr_higher, _ = roc_curve(y_true, losses)
    advantage_higher = float(np.max(tpr_higher - fpr_higher))

    if auc_lower >= auc_higher:
        best_auc, best_advantage, best_direction = auc_lower, advantage_lower, "lower_loss_is_member"
    else:
        best_auc, best_advantage, best_direction = auc_higher, advantage_higher, "higher_loss_is_member"

    return {
        "auc": best_auc,
        "advantage": best_advantage,
        "attack_direction": best_direction,
        "auc_lower_loss_is_member": auc_lower,
        "auc_higher_loss_is_member": auc_higher,
        "mean_member_loss": float(np.mean(member_losses)),
        "mean_non_member_loss": float(np.mean(non_member_losses)),
        "num_member": int(len(member_losses)),
        "num_non_member": int(len(non_member_losses)),
    }
