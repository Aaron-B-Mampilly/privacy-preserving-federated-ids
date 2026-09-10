"""Phase 5: the Flower client wrapping the Phase 4 model + training code.

FedAvg averages weight TENSORS across clients, which is only valid if
every participating client's model has IDENTICAL architecture --
including the classifier's output size. Phase 4's local-only baseline
deliberately let each client have its own class count C_k (correct
there, since local models were never aggregated); Phase 5 has no
personalization yet (that's Phase 6 -- the whole model, encoder +
decoder + classifier, is FedAveraged together), so every client here
MUST share one global class vocabulary. See
sequence_dataset.build_scope_dataloaders's `label_to_index` override
and federated/simulation.py, which builds that shared vocabulary once
and hands it to every client.

Local epochs use a FRESH optimizer each round (no momentum state
carried across rounds) -- this is standard FedAvg: only model weights
travel between server and client, never optimizer state.
"""

from __future__ import annotations

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from torch.utils.data import DataLoader

from fedpda_ids.models.trainer import evaluate, train_one_epoch


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


class FlowerLSTMClient(NumPyClient):
    """One federated client. Ephemeral per Flower's simulation model --
    holds a reference to this client's own train/val DataLoaders (never
    another client's, enforced by how simulation.py constructs them)
    and trains/evaluates whatever global weights it's handed each round.
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

    def get_parameters(self, config) -> list[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config) -> tuple[list[np.ndarray], int, dict]:
        set_model_parameters(self.model, parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        last_metrics = {}
        for _ in range(self.local_epochs):
            last_metrics = train_one_epoch(self.model, self.train_loader, optimizer, self.device, self.lambda_ce)

        num_examples = len(self.train_loader.dataset)
        return get_model_parameters(self.model), num_examples, {"train_loss": last_metrics.get("loss", float("nan"))}

    def evaluate(self, parameters: list[np.ndarray], config) -> tuple[float, int, dict]:
        set_model_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.val_loader, self.device, self.lambda_ce, self.index_to_label)
        num_examples = len(self.val_loader.dataset)
        return float(metrics["loss"]), num_examples, {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }


def client_to_flower_client(flower_lstm_client: FlowerLSTMClient) -> Client:
    return flower_lstm_client.to_client()
