"""Phase 5/6: the Flower client wrapping the Phase 4 model + training code.

Phase 5 (vanilla FedAvg): FedAvg averages weight TENSORS across
clients, which is only valid if every participating client's model
has IDENTICAL architecture -- including the classifier's output size.
Phase 4's local-only baseline deliberately let each client have its
own class count C_k (correct there, since local models were never
aggregated); Phase 5 has no personalization, so every client shares
one global class vocabulary (see sequence_dataset's label_to_index
override).

Phase 6 (personalization): ONLY encoder+decoder ("shared") parameters
travel to/from the server; the classifier head ("local") never does.
This raises a real problem Phase 5 didn't have: Flower's simulation
clients are EPHEMERAL -- client_fn constructs a brand-new model
(random init) on every single invocation, and Flower's own docs say
client instances "should not attempt to carry state over method
invocations." Ray's actor pool doesn't guarantee the same client_id
lands on the same worker twice, so an in-memory cache would silently
fail. The only robust fix is disk-based persistence: each client's
local head is saved to `{checkpoint_dir}/personalized_heads/{run_name}
/client_{id}.pt` at the end of every fit() call and reloaded at
client construction time -- correct regardless of which process
handles which call, and regardless of whether Flower gives fit() and
evaluate() the same or different underlying Python objects within one
round (verified this is NOT guaranteed).

Local epochs use a FRESH optimizer each round (no momentum state
carried across rounds) -- this is standard FedAvg: only model weights
travel between server and client, never optimizer state. (Phase 6:
the optimizer still updates ALL parameters, including the local head,
during local training -- only what's COMMUNICATED is restricted to
the shared subset.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from torch.utils.data import DataLoader

from fedpda_ids.models.trainer import evaluate, train_one_epoch
from fedpda_ids.privacy.dp import clip_update, compute_update

SHARED_PREFIXES = ("encoder.", "decoder.")


def get_model_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    # .copy(): .cpu().numpy() aliases the tensor's live memory. Without
    # copying, any later load_state_dict()/optimizer.step() on this
    # model (e.g. inside fit(), called with these SAME extracted
    # parameters as input) silently mutates the values a caller thinks
    # it already captured -- found via test_client.py's fit() tests.
    return [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]


def set_model_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


def get_shared_parameter_keys(model: torch.nn.Module) -> list[str]:
    return [k for k in model.state_dict().keys() if k.startswith(SHARED_PREFIXES)]


def get_shared_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    state_dict = model.state_dict()
    return [state_dict[k].detach().cpu().numpy().copy() for k in get_shared_parameter_keys(model)]


def set_shared_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    """Merges ONLY the encoder/decoder tensors into the model --
    strict=False so the (absent) classifier.* keys are left exactly as
    they already are (the client's own locally-trained head)."""
    keys = get_shared_parameter_keys(model)
    assert len(keys) == len(parameters), f"expected {len(keys)} shared tensors, got {len(parameters)}"
    partial_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(partial_state, strict=False)


def local_head_path(checkpoint_dir: str | Path, run_name: str, client_id: int) -> Path:
    return Path(checkpoint_dir) / "personalized_heads" / run_name / f"client_{client_id}.pt"


def load_local_head(model: torch.nn.Module, path: Path) -> bool:
    """Returns True if a persisted head was found and loaded, False if
    this client has never been sampled before (model keeps whatever
    fresh random classifier weights it was constructed with)."""
    if not path.exists():
        return False
    classifier_state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(classifier_state, strict=False)
    return True


def save_local_head(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classifier_state = {k: v for k, v in model.state_dict().items() if k.startswith("classifier.")}
    torch.save(classifier_state, path)


class FlowerLSTMClient(NumPyClient):
    """One federated client. Ephemeral per Flower's simulation model --
    holds a reference to this client's own train/val DataLoaders (never
    another client's, enforced by how simulation.py constructs them)
    and trains/evaluates whatever global weights it's handed each round.

    `personalized`: when True, only encoder/decoder parameters are
    exchanged with the server (Phase 6); the classifier head is
    persisted to `head_path` across rounds instead. When False (Phase
    5 behavior, default), the full model is exchanged, unchanged from
    before.

    `proximal_mu` (E1's FedProx comparator, default 0.0 = plain FedAvg,
    unchanged): adds a proximal term anchoring local training to this
    round's starting (global) parameters -- see train_one_epoch's
    docstring. Only meaningful with personalized=False (FedProx is a
    full-model-exchange baseline, evaluated on the same basis as
    vanilla FedAvg, not combined with personalization).

    `dp_clip_norm` (E6's DP+SecAgg+ combined comparator, default None =
    unchanged behavior): when set, fit() returns this round's CLIPPED
    UPDATE (shared params after local training minus what this client
    started the round with, L2-clipped to this FIXED public constant)
    instead of raw parameters, and reports num_examples=1 regardless of
    this client's real dataset size. Both changes exist so Flower's
    SecAgg+ workflow -- which must sum/average client values without
    ever seeing any individual one -- can be given a value it's safe to
    combine: a fixed (not per-round-adaptive) clip bound needs no
    server-side visibility into individual norms, and forcing
    num_examples=1 for every client makes SecAgg+'s internal per-client
    weighting ratio identical across clients, so the value it reveals
    is the plain, UNWEIGHTED mean of clipped deltas -- this project's
    established uniform client-level DP-FedAvg convention (see
    src/fedpda_ids/privacy/dp.py's module docstring). Only meaningful
    with personalized=True (DP protects the shared encoder/decoder
    only, per dp.py's docstring; the local classifier head never
    leaves the client and needs no clipping).
    """

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int,
        lambda_ce: float,
        learning_rate: float,
        index_to_label: dict[int, str],
        personalized: bool = False,
        head_path: Path | None = None,
        proximal_mu: float = 0.0,
        dp_clip_norm: float | None = None,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lambda_ce = lambda_ce
        self.learning_rate = learning_rate
        self.index_to_label = index_to_label
        self.personalized = personalized
        self.head_path = head_path
        self.proximal_mu = proximal_mu
        self.dp_clip_norm = dp_clip_norm

        if self.personalized:
            assert self.head_path is not None, "personalized=True requires head_path"
            load_local_head(self.model, self.head_path)
        if self.dp_clip_norm is not None:
            assert self.personalized, "dp_clip_norm requires personalized=True (DP protects shared params only)"

    def get_parameters(self, config) -> list[np.ndarray]:
        if self.personalized:
            return get_shared_parameters(self.model)
        return get_model_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config) -> tuple[list[np.ndarray], int, dict]:
        if self.personalized:
            set_shared_parameters(self.model, parameters)
        else:
            set_model_parameters(self.model, parameters)

        # dp_clip_norm mode needs the round's STARTING shared params to
        # compute this client's update after training -- `parameters` is
        # exactly that (the values just loaded above, before any local
        # training touches them), so no extra model round-trip is needed.
        round_start_shared_params = parameters if self.dp_clip_norm is not None else None

        global_params = (
            [p.detach().clone() for p in self.model.parameters()] if self.proximal_mu > 0.0 else None
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        last_metrics = {}
        for _ in range(self.local_epochs):
            last_metrics = train_one_epoch(
                self.model, self.train_loader, optimizer, self.device, self.lambda_ce,
                proximal_mu=self.proximal_mu, global_params=global_params,
            )

        if self.personalized:
            save_local_head(self.model, self.head_path)
            returned_params = get_shared_parameters(self.model)
        else:
            returned_params = get_model_parameters(self.model)

        num_examples = len(self.train_loader.dataset)

        if self.dp_clip_norm is not None:
            update = compute_update(returned_params, round_start_shared_params)
            returned_params = clip_update(update, self.dp_clip_norm)
            num_examples = 1  # forces SecAgg+'s per-client weighting ratio to be identical for every client

        return returned_params, num_examples, {"train_loss": last_metrics.get("loss", float("nan"))}

    def evaluate(self, parameters: list[np.ndarray], config) -> tuple[float, int, dict]:
        if self.personalized:
            set_shared_parameters(self.model, parameters)
            # local head was already loaded in __init__ (this client
            # instance's own most recent persisted head, whether from
            # this round's fit or an earlier one)
        else:
            set_model_parameters(self.model, parameters)

        metrics = evaluate(self.model, self.val_loader, self.device, self.lambda_ce, self.index_to_label)
        num_examples = len(self.val_loader.dataset)
        return float(metrics["loss"]), num_examples, {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mse": metrics["mse"],
        }


def client_to_flower_client(flower_lstm_client: FlowerLSTMClient) -> Client:
    return flower_lstm_client.to_client()
