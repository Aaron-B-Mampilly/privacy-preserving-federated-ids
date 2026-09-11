"""Phase 7: latent class prototypes for rare-class/zero-day detection.

Built on top of an already-trained shared encoder (Phase 5/6's FedAvg
or personalized-FL output) -- this is a post-training mechanism, not
part of the per-round FL loop. The server never receives raw examples,
only clipped (and, from Phase 8 onward, noised) per-class latent means.

DP noise is explicitly DEFERRED to Phase 8 (see Phase 7 kickoff
PRE-CODING CORRECTION): `client_prototypes()` accepts an optional
`noise_fn` applied AFTER clipping, defaulting to a no-op (equivalent
to epsilon=infinity for this mechanism specifically). Phase 8 plugs in
calibrated Gaussian noise here without restructuring anything.

Decision mechanism, exactly per the frozen spec:
    1. distance(z, every global prototype)
    2. min distance > threshold  ->  NEW CLASS (zero-day)
    3. else                      ->  nearest prototype's label
       (this IS the "prototype-based rare-class transfer" path when
       that label isn't one of the querying client's own local classes
       -- the caller decides whether to prefer its local head's
       prediction for locally-known classes; classify_with_prototypes()
       itself always returns the nearest-prototype answer, since it
       has no notion of "local classes" -- that's scope-specific).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_latents(encoder: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Runs ONLY the encoder (no decoder/classifier) over a DataLoader,
    collecting (latent, label) pairs. Labels are whatever the loader's
    second tensor element is -- an integer class index."""
    encoder.eval()
    all_z, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x)
        all_z.append(z.cpu().numpy())
        all_y.append(y.numpy())
    if not all_z:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_z), np.concatenate(all_y)


def compute_class_prototypes(z: np.ndarray, labels: np.ndarray) -> dict[int, tuple[np.ndarray, int]]:
    """Mean latent vector per class label present in (z, labels).
    Returns {label: (prototype_vector, support_count)} -- the count is
    kept alongside the vector since aggregation and threshold
    calibration both need per-class weights."""
    prototypes = {}
    for label in np.unique(labels):
        mask = labels == label
        prototypes[int(label)] = (z[mask].mean(axis=0), int(mask.sum()))
    return prototypes


def clip_prototype(vector: np.ndarray, bound: float = 1.0) -> np.ndarray:
    """L2-norm clip to `bound` -- the frozen, FIXED prototype clipping
    bound B=1.0 (distinct from Phase 8's planned ADAPTIVE model-update
    clipping, which is median-based and not fixed)."""
    norm = np.linalg.norm(vector)
    if norm <= bound or norm == 0:
        return vector.copy()
    return vector * (bound / norm)


def client_prototypes(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    clip_bound: float = 1.0,
    noise_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[int, tuple[np.ndarray, int]]:
    """One client's protected per-class prototypes: extract -> mean ->
    clip -> (optionally) noise. This is exactly what gets transmitted
    to the server -- never raw examples, never unclipped vectors."""
    z, labels = extract_latents(encoder, loader, device)
    if len(labels) == 0:
        return {}
    raw = compute_class_prototypes(z, labels)
    protected = {}
    for label, (vector, count) in raw.items():
        clipped = clip_prototype(vector, clip_bound)
        if noise_fn is not None:
            clipped = noise_fn(clipped)
        protected[label] = (clipped, count)
    return protected


def aggregate_prototypes(
    client_prototype_list: list[dict[int, tuple[np.ndarray, int]]],
    return_support: bool = False,
):
    """Server-side aggregation: for each class, a support-weighted mean
    across every client that reported a (protected) prototype for it.
    Never touches raw examples -- only the already-clipped/noised
    vectors each client sent.

    `return_support=True` additionally returns each class's total
    support (summed sequence count across contributing clients) --
    needed as calibrate_threshold()'s class_weights argument by the
    real pipeline. Default False keeps the original single-dict return
    every existing caller/test already relies on."""
    sums: dict[int, np.ndarray] = {}
    weights: dict[int, int] = {}
    for client_protos in client_prototype_list:
        for label, (vector, count) in client_protos.items():
            if label not in sums:
                sums[label] = vector * count
                weights[label] = count
            else:
                sums[label] = sums[label] + vector * count
                weights[label] += count
    aggregated = {label: sums[label] / weights[label] for label in sums}
    if return_support:
        return aggregated, weights
    return aggregated


def calibrate_threshold(
    global_prototypes: dict[int, np.ndarray],
    class_weights: dict[int, int],
    multiplier: float = 0.5,
) -> float:
    """tau = (support-weighted mean pairwise distance between global
    prototypes) * multiplier. Must be called with prototypes/weights
    derived from TRAIN+VAL data only, then frozen -- never recomputed
    against test/zero-day data (the caller is responsible for that
    split discipline; this function just does the arithmetic)."""
    labels = sorted(global_prototypes.keys())
    if len(labels) < 2:
        raise ValueError("need >=2 classes to calibrate a pairwise-distance threshold")

    weighted_sum = 0.0
    weight_sum = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            dist = float(np.linalg.norm(global_prototypes[li] - global_prototypes[lj]))
            w = class_weights.get(li, 1) * class_weights.get(lj, 1)
            weighted_sum += w * dist
            weight_sum += w

    mean_pairwise_distance = weighted_sum / weight_sum
    return mean_pairwise_distance * multiplier


def classify_with_prototypes(
    z: np.ndarray, global_prototypes: dict[int, np.ndarray], threshold: float, clip_bound: float | None = None
) -> tuple[int | None, float]:
    """Returns (predicted_label_or_None, min_distance). None means NEW
    CLASS (zero-day) -- min_distance exceeded threshold.

    `clip_bound`: PRE-CODING CORRECTION (Phase 7 real-data pipeline,
    approved by user) -- prototypes are L2-clipped to B at training
    time (privacy protection for the transmitted statistic), but an
    encoder's natural latent norm can differ arbitrarily from B (e.g.
    ~2.28 vs B=1.0 on real CICIDS2017 data), which would otherwise put
    every query point at a near-constant, uninformative distance from
    every prototype. Passing the same B here symmetrically clips the
    query latent before distance comparison, so both sides of the
    comparison live on the same scale. Default None preserves the
    original (unclipped-query) behavior this function was first tested
    with -- callers must opt in explicitly."""
    if clip_bound is not None:
        z = clip_prototype(z, clip_bound)
    best_label, best_dist = None, float("inf")
    for label, prototype in global_prototypes.items():
        dist = float(np.linalg.norm(z - prototype))
        if dist < best_dist:
            best_dist, best_label = dist, label
    if best_dist > threshold:
        return None, best_dist
    return best_label, best_dist


def classify_batch_with_prototypes(
    z_batch: np.ndarray, global_prototypes: dict[int, np.ndarray], threshold: float, clip_bound: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of classify_with_prototypes for a batch of
    latents. Returns (predictions, min_distances); predictions use -1
    as the NEW CLASS sentinel (consistent with this project's existing
    -1-means-"not a real client/class" convention elsewhere).

    `clip_bound`: see classify_with_prototypes docstring -- same
    symmetric-clip opt-in, vectorized here (equivalent to applying
    clip_prototype() to every row)."""
    if clip_bound is not None:
        norms = np.linalg.norm(z_batch, axis=1, keepdims=True)
        scale = np.where(norms > clip_bound, clip_bound / np.where(norms == 0, 1.0, norms), 1.0)
        z_batch = z_batch * scale
    labels = sorted(global_prototypes.keys())
    proto_matrix = np.stack([global_prototypes[label] for label in labels])  # (num_classes, latent_dim)
    dists = np.linalg.norm(z_batch[:, None, :] - proto_matrix[None, :, :], axis=2)  # (batch, num_classes)
    nearest_idx = dists.argmin(axis=1)
    min_dists = dists[np.arange(len(z_batch)), nearest_idx]
    predictions = np.array([labels[i] for i in nearest_idx])
    predictions = np.where(min_dists > threshold, -1, predictions)
    return predictions, min_dists
