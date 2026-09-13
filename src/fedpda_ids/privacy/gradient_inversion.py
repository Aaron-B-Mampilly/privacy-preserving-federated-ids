"""E6's gradient inversion attack (DLG -- Zhu et al. 2019, "Deep
Leakage from Gradients") -- the standard way to empirically test
whether an observed FL update lets an attacker (a curious server)
reconstruct a client's real INPUT, independent of Phase 11's
loss-threshold MIA (which only tests membership, never content,
leakage).

Mechanism: given a REAL gradient a client would send for one local
step, initialize a random "dummy" input (and dummy soft label), then
optimize BOTH by gradient descent so the dummy input's OWN gradient
(through the SAME model, via autograd's double-backward) matches the
real gradient being attacked as closely as possible. If the dummy
gradient can be made to match, the dummy input converges toward the
real one (Zhu et al.'s core empirical result). Reconstruction quality
is reported as MSE(real_x, dummy_x) -- LOWER means MORE leaked (this is
a privacy-risk score, not a utility score -- do not read it as "higher
is better").

Attacked object, matching E6's Table T6's three comparators exactly:
- "FedAvg updates": the RAW gradient of one client's real local batch,
  with NO protection -- the worst case (highest expected leakage), what
  a curious FedAvg server would see directly. Attacks the FULL model
  (Phase 5's FlowerLSTMClient exchanges every parameter, personalized=False).
- "Ours+DP": the SAME gradient after Phase 8's OWN clip-then-noise
  treatment (reusing clip_update/add_gaussian_noise from dp.py) --
  still an INDIVIDUAL client's own (now-protected) contribution, since
  DP alone never hides WHICH client sent it, only degrades its content.
  Attacks only the SHARED encoder/decoder parameters (personalized FL's
  transmitted subset -- the local classifier head never leaves the
  client and is never attackable).
- "Ours+DP+SecAgg": the attacker never sees any individual client's
  contribution at all under SecAgg+ (that is precisely what it's for)
  -- only the MEAN across `clients_per_round` clients' clip+noise
  contributions (mirroring item 4's DPSecAggPersonalizedFedAvg mechanism
  exactly). The attack still targets ONE specific client's real input,
  but only has this m-client average to work with.

This project's local training uses multiple epochs/minibatches of Adam
(not a single SGD step), so "the gradient a client would send" here is
defined, per the standard DLG evaluation methodology used across the FL
privacy literature, as the gradient of ONE real local minibatch at the
CURRENT (checkpointed) shared parameters -- the single-step case is the
standard, most-attackable reference point; a real multi-epoch update
only makes the attacker's job harder, so this is the conservative
(strongest-attack, not inflated) comparison point.

Batch size is a genuine, documented limitation shared with the entire
DLG literature: reconstruction quality degrades sharply as batch size
grows past a handful of examples (more unknowns than gradient
equations to match them against) -- callers should attack small batches
(1-2 examples), not a full training minibatch, and this is a property
of the attack itself, not a bug in this implementation.
"""

from __future__ import annotations

import torch
from torch import nn

from fedpda_ids.models.lstm_autoencoder import compute_total_loss


def compute_client_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_ce: float,
    shared_only_prefixes: tuple[str, ...] | None = None,
) -> list[torch.Tensor]:
    """The gradient of one real local minibatch -- what DLG attacks.

    `shared_only_prefixes` (e.g. `client.SHARED_PREFIXES`): restricts
    the attacked gradient to the SAME parameter subset personalized FL
    actually transmits. `model.eval()` is required here (not just a
    style choice): dropout in the classifier head would otherwise make
    two forward passes at the SAME input/parameters return DIFFERENT
    gradients, breaking DLG's core assumption that the target gradient
    is a deterministic function of (params, x, y).
    """
    model.eval()
    model.zero_grad()
    reconstruction, logits, _latent = model(x)
    loss, _mse, _ce = compute_total_loss(reconstruction, x, logits, y, lambda_ce)
    params = [p for name, p in model.named_parameters() if shared_only_prefixes is None or name.startswith(shared_only_prefixes)]
    grads = torch.autograd.grad(loss, params, create_graph=False)
    return [g.detach().clone() for g in grads]


def aggregate_gradients(grad_list: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Mean of m clients' gradients -- what SecAgg+ reveals to the
    server instead of any individual client's contribution (the exact
    mean DPSecAggPersonalizedFedAvg's aggregate_fit reconstructs from,
    see simulation.py)."""
    m = len(grad_list)
    num_tensors = len(grad_list[0])
    return [sum(g[i] for g in grad_list) / m for i in range(num_tensors)]


def dlg_reconstruct(
    model: nn.Module,
    target_grad: list[torch.Tensor],
    x_shape: tuple[int, ...],
    num_classes: int,
    lambda_ce: float,
    shared_only_prefixes: tuple[str, ...] | None = None,
    num_iterations: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> dict:
    """Zhu et al.'s DLG optimization loop: recover a dummy (x, soft-label
    y) pair whose OWN gradient (through the SAME model/parameters)
    matches `target_grad` as closely as possible. Uses L-BFGS (the
    original paper's own optimizer choice -- second-order, converges in
    far fewer steps than first-order SGD/Adam for this kind of gradient-
    matching problem) with `max_iter=1` per outer step so
    `grad_distance_history` gets one entry per real optimizer step.
    """
    torch.manual_seed(seed)
    dummy_x = torch.randn(x_shape, requires_grad=True)
    dummy_y_logits = torch.randn(x_shape[0], num_classes, requires_grad=True)

    model.eval()
    params = [p for name, p in model.named_parameters() if shared_only_prefixes is None or name.startswith(shared_only_prefixes)]

    optimizer = torch.optim.LBFGS([dummy_x, dummy_y_logits], lr=lr, max_iter=1)
    grad_distance_history: list[float] = []

    def closure():
        optimizer.zero_grad()
        dummy_y_soft = torch.softmax(dummy_y_logits, dim=-1)
        reconstruction, logits, _latent = model(dummy_x)
        mse = nn.functional.mse_loss(reconstruction, dummy_x)
        log_probs = torch.log_softmax(logits, dim=-1)
        ce = -(dummy_y_soft * log_probs).sum(dim=-1).mean()
        dummy_loss = mse + lambda_ce * ce
        dummy_grad = torch.autograd.grad(dummy_loss, params, create_graph=True)
        grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grad, target_grad))
        grad_diff.backward()
        return grad_diff

    for _ in range(num_iterations):
        grad_diff = optimizer.step(closure)
        grad_distance_history.append(float(grad_diff.item()))

    return {
        "dummy_x": dummy_x.detach(),
        "dummy_y_logits": dummy_y_logits.detach(),
        "grad_distance_history": grad_distance_history,
        "final_grad_distance": grad_distance_history[-1] if grad_distance_history else float("nan"),
    }


def reconstruction_mse(real_x: torch.Tensor, dummy_x: torch.Tensor) -> float:
    """The reconstruction-quality metric E6's Table T6 reports -- LOWER
    is MORE leakage (a privacy-risk score), the opposite convention from
    every other MSE in this project (which is a utility metric)."""
    return float(torch.mean((real_x - dummy_x) ** 2).item())
