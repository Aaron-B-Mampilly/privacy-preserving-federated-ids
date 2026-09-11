"""Phase 8: client-level differential privacy for the federated model
updates exchanged in Phase 6's personalized FL (the shared encoder/
decoder only -- the local classifier head never leaves the client and
needs no protection here; Phase 7's prototype vectors get their own,
separate clip-and-noise treatment via client_prototypes()'s noise_fn).

Mechanism, exactly per the frozen spec:
    1. Each participating client's UPDATE (this round's trained shared
       params minus what it started the round with) is L2-clipped to
       C_t, an ADAPTIVE threshold -- the MEDIAN of this round's actual
       client update norms (not a fixed constant).
    2. Clipped updates are summed, and ONE Gaussian noise vector
       (std = noise_multiplier * C_t) is added to that sum -- the
       sensitivity of the sum (how much one client's presence/absence
       can change it) is exactly C_t, which is what noise_multiplier
       is calibrated against.
    3. The noised sum is divided by m = clients_per_round, a FIXED
       config-level constant -- NOT each client's own (private)
       dataset size. See Phase 8 kickoff PRE-CODING CORRECTION (user-
       approved): standard client-level DP-FedAvg (McMahan et al.,
       2018) requires uniform per-client weighting for the accountant's
       sensitivity bound to hold; this deliberately overrides the base
       FedAvg policy of weighting by dataset size, for DP runs only --
       Phase 5/6's own non-DP official results are untouched.

Calibrating noise_multiplier from a target (epsilon, delta) uses
Opacus's RDP accountant, treating one FL round of client subsampling as
one "step" of a subsampled Gaussian mechanism -- exactly the DP-FedAvg
construction this accountant's math is built for. epsilon=inf is the
"no DP" sentinel throughout this module (noise_multiplier=0.0, meaning
the Phase 6 result IS the epsilon=inf point of the sweep -- no
separate run needed).
"""

from __future__ import annotations

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier as _opacus_get_noise_multiplier


def compute_update(new_params: list[np.ndarray], old_params: list[np.ndarray]) -> list[np.ndarray]:
    """This round's client update: new (post-local-training) shared
    params minus what the client started the round with."""
    return [new - old for new, old in zip(new_params, old_params)]


def update_norm(update: list[np.ndarray]) -> float:
    """ONE L2 norm across every parameter array in the update, treated
    as a single flattened vector -- not one norm per layer."""
    return float(np.sqrt(sum(np.sum(np.square(arr)) for arr in update)))


def adaptive_clip_threshold(update_norms: list[float]) -> float:
    """C_t = median of this round's actual client update norms."""
    if not update_norms:
        raise ValueError("need at least one client update norm to compute a median clip threshold")
    return float(np.median(update_norms))


def clip_update(update: list[np.ndarray], clip_norm: float) -> list[np.ndarray]:
    """L2-clip the WHOLE update (every array together) to clip_norm --
    same scale factor applied to every array, direction preserved."""
    norm = update_norm(update)
    if norm <= clip_norm or norm == 0:
        return [arr.copy() for arr in update]
    scale = clip_norm / norm
    return [arr * scale for arr in update]


def add_gaussian_noise(
    summed_update: list[np.ndarray], noise_multiplier: float, clip_norm: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Adds one noise vector (std = noise_multiplier * clip_norm) to
    the SUM of clipped updates. noise_multiplier=0.0 is the "no DP"
    sentinel -- returns an unchanged copy, never a degenerate zero-std
    normal draw."""
    if noise_multiplier == 0.0:
        return [arr.copy() for arr in summed_update]
    std = noise_multiplier * clip_norm
    return [arr + rng.normal(0.0, std, size=arr.shape).astype(arr.dtype) for arr in summed_update]


def calibrate_noise_multiplier(
    target_epsilon: float, target_delta: float, sample_rate: float, num_rounds: int
) -> float:
    """target_epsilon=inf -> 0.0 (no DP). Any finite epsilon is
    calibrated via Opacus's RDP accountant: sample_rate =
    clients_per_round / pool_size (this round's client-subsampling
    probability), num_rounds = total FL rounds (one accountant "step"
    per round)."""
    if target_epsilon == float("inf"):
        return 0.0
    return _opacus_get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=num_rounds,
        accountant="rdp",
    )


def compute_achieved_epsilon(noise_multiplier: float, sample_rate: float, num_rounds: int, delta: float) -> float:
    """Verifies what epsilon a completed run ACTUALLY spent, rather
    than trusting the calibration target blindly -- reported alongside
    every DP experiment's results. noise_multiplier=0.0 -> inf."""
    if noise_multiplier == 0.0:
        return float("inf")
    accountant = RDPAccountant()
    for _ in range(num_rounds):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(accountant.get_epsilon(delta=delta))


def calibrate_prototype_noise_std(
    target_epsilon: float, target_delta: float, clip_bound: float, num_classes: int
) -> float:
    """Phase 7's prototype noise_fn placeholder, filled in for Phase 8.

    Design decision (genuine spec gap -- the frozen spec says "clipped
    (B=1.0) + DP noise before transmission" but doesn't say how to
    account for a client contributing to MULTIPLE classes at once):
    under CLIENT-level adjacency (this project's mechanism throughout,
    not per-example), removing one client can change every one of the
    (up to num_classes) per-class prototype sums it contributed to
    simultaneously -- so the concatenated release across all classes
    has JOINT L2 sensitivity clip_bound * sqrt(num_classes) in the
    worst case (a client present in every class), not clip_bound alone.
    Composing k independent Gaussian-mechanism releases that each use
    the SAME noise std sigma is Renyi-DP-equivalent to one release with
    sensitivity scaled by sqrt(k) at that same sigma (Renyi divergence
    adds linearly under composition) -- so calibrating sigma against
    the single-release Gaussian mechanism (sample_rate=1.0, one
    "round" -- every trainable client contributes its prototypes
    exactly once, not a per-round subsample) with sensitivity
    clip_bound * sqrt(num_classes), then using that SAME sigma for
    every class's noise draw, is the correct, non-naive way to keep the
    joint release within the target (epsilon, delta) budget."""
    if target_epsilon == float("inf"):
        return 0.0
    noise_multiplier = calibrate_noise_multiplier(target_epsilon, target_delta, sample_rate=1.0, num_rounds=1)
    joint_sensitivity = clip_bound * float(np.sqrt(num_classes))
    return noise_multiplier * joint_sensitivity
