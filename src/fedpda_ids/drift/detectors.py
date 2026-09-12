"""Phase 10: unsupervised concept-drift detection on the trained shared
encoder/decoder's reconstruction error and latent representation.

Frozen spec, exactly: ADWIN (delta=0.002) on reconstruction error is the
PRIMARY detector; PSI (threshold=0.25) on latent dimensions, calibrated
against a reference of the last 5000 BENIGN windows, is the BACKUP
detector. Detection is entirely UNSUPERVISED -- neither detector ever
looks at ground-truth labels, only reconstruction error / latent
vectors the encoder/decoder already produce.

Two genuine spec gaps, documented here rather than silently picked:
    1. PSI is a per-variable statistic, but the latent representation
       is 32-dimensional. This module computes PSI independently per
       latent dimension, then reports the MEAN across dimensions as
       "the" PSI score for a window -- consistent with the retrain-
       trigger config's own "mean drift" phrasing.
    2. "mean drift > threshold for N consecutive rounds" (the retrain
       trigger) is computed by chunking a chronologically-ordered
       stream into fixed-size ROUNDS, and for each round taking the
       mean of two binary per-round flags: "did ADWIN signal drift
       anywhere in this round" and "did this round's mean latent PSI
       exceed psi_threshold". mean_drift in {0.0, 0.5, 1.0}; the
       trigger fires when mean_drift >= 0.5 (at least one of the two
       detectors flags that round) for `consecutive_rounds` rounds in
       a row -- i.e. either detector alone is sufficient per round,
       matching a primary+backup (not requires-both-to-agree) design.

Retraining itself is explicitly OUT of this module's scope (user
decision, Phase 10 kickoff) -- these functions DETECT and REPORT where
the trigger condition would fire; actually re-training on trigger is
deferred to Phase 12's full experiment suite.

`round_window_size` must be large enough for PSI to be statistically
stable -- verified empirically (test_detectors.py): with num_bins=10,
a round of 50 samples (~5/bin) lets PSI spuriously exceed the 0.25
threshold on PURE NOISE alone (no real drift), while >=200 samples/
round stays comfortably below threshold under the null. Callers should
use round_window_size >= 200-500, not an arbitrarily small chunk size.
"""

from __future__ import annotations

import numpy as np
import torch
from river.drift import ADWIN
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_per_sequence_errors_and_latents(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the full autoencoder (encoder+decoder) over a DataLoader,
    returning (per_sequence_mse, latents) -- one reconstruction-MSE
    scalar and one latent vector PER SEQUENCE (never batch-averaged;
    ADWIN needs a stream of individual values, not one number per batch)."""
    model.eval()
    all_errors, all_latents = [], []
    for x, _y in loader:
        x = x.to(device)
        reconstruction, _logits, latent = model(x)
        per_sample_mse = torch.mean((reconstruction - x) ** 2, dim=tuple(range(1, x.dim())))
        all_errors.append(per_sample_mse.cpu().numpy())
        all_latents.append(latent.cpu().numpy())
    if not all_errors:
        return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_errors), np.concatenate(all_latents)


def run_adwin_detector(errors: np.ndarray, delta: float = 0.002) -> list[int]:
    """Feeds a reconstruction-error stream through river's ADWIN,
    IN ORDER (caller must have already sorted `errors` chronologically).
    Returns the indices at which ADWIN signaled a change point."""
    detector = ADWIN(delta=delta)
    drift_points = []
    for i, value in enumerate(errors):
        detector.update(float(value))
        if detector.drift_detected:
            drift_points.append(i)
    return drift_points


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Population Stability Index for ONE 1-D distribution: bin edges
    from the REFERENCE distribution's own quantiles (standard PSI
    convention), compare reference vs current bin proportions.
    PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)).
    A small epsilon avoids log(0)/div-by-0 for empty bins."""
    eps = 1e-6
    quantile_edges = np.unique(np.quantile(reference, np.linspace(0, 1, num_bins + 1)))
    if len(quantile_edges) < 3:
        # reference is (near-)constant -- can't form meaningful bins
        return 0.0
    edges = quantile_edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(len(reference), 1) + eps
    cur_pct = cur_counts / max(len(current), 1) + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_latent_psi(reference_latents: np.ndarray, current_latents: np.ndarray, num_bins: int = 10) -> float:
    """Mean PSI across every latent dimension (see module docstring,
    design gap #1) -- reference_latents/current_latents: (N, latent_dim)."""
    num_dims = reference_latents.shape[1]
    per_dim_psi = [
        compute_psi(reference_latents[:, d], current_latents[:, d], num_bins=num_bins)
        for d in range(num_dims)
    ]
    return float(np.mean(per_dim_psi))


def evaluate_drift_stream(
    errors: np.ndarray,
    latents: np.ndarray,
    reference_latents: np.ndarray,
    round_window_size: int,
    adwin_delta: float = 0.002,
    psi_threshold: float = 0.25,
    consecutive_rounds_for_retrain: int = 2,
    psi_num_bins: int = 10,
) -> dict:
    """High-level replay: chunks a CHRONOLOGICALLY-ORDERED stream
    (errors + latents, same order/length) into fixed-size rounds,
    runs both detectors per round, and reports where the
    consecutive-rounds retrain trigger would fire. See module
    docstring, design gap #2, for exactly how "mean drift" and the
    trigger condition are defined.

    `reference_latents`: the fixed PSI baseline (spec: last 5000
    BENIGN windows) -- computed ONCE by the caller from designated
    calibration data, never updated as the stream is replayed."""
    n = len(errors)
    adwin_drift_indices = run_adwin_detector(errors, delta=adwin_delta)
    adwin_drift_set = set(adwin_drift_indices)

    round_reports = []
    consecutive = 0
    first_trigger_round = None
    for round_idx, start in enumerate(range(0, n, round_window_size)):
        end = min(start + round_window_size, n)
        adwin_flag = any(start <= i < end for i in adwin_drift_set)
        round_psi = compute_latent_psi(reference_latents, latents[start:end], num_bins=psi_num_bins)
        psi_flag = round_psi > psi_threshold
        mean_drift = float(np.mean([adwin_flag, psi_flag]))

        if mean_drift >= 0.5:
            consecutive += 1
        else:
            consecutive = 0

        triggered = consecutive >= consecutive_rounds_for_retrain
        if triggered and first_trigger_round is None:
            first_trigger_round = round_idx

        round_reports.append({
            "round": round_idx, "start": start, "end": end,
            "adwin_flag": adwin_flag, "psi": round_psi, "psi_flag": psi_flag,
            "mean_drift": mean_drift, "consecutive_rounds_over_threshold": consecutive,
            "retrain_triggered": triggered,
        })

    return {
        "num_sequences": n,
        "num_rounds": len(round_reports),
        "adwin_drift_indices": adwin_drift_indices,
        "first_trigger_round": first_trigger_round,
        "round_reports": round_reports,
    }
