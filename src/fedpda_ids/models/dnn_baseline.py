"""Phase 4 Part J: minimal LSTM-vs-DNN ablation.

Research question: does genuine temporal modeling (the LSTM encoder,
which sees all 10 timesteps in order) improve intrusion detection
over a non-temporal baseline that only sees the CURRENT observation
(the window's last row -- the same row the sequence's label is
defined from)? This isolates exactly the LSTM's temporal value-add,
rather than comparing against a DNN that secretly still sees the
whole window (which wouldn't be "non-temporal" at all).

Comparable parameter scale to the frozen encoder + classification
head: Linear(F->64) -> ReLU -> Linear(64->32) -> ReLU, then the SAME
classification head shape (Linear(32->32)->ReLU->Dropout->Linear(32->C)).

Scope note: this is classification-only. The frozen model's
reconstruction term operates over a 10-step window; reconstructing a
single timestep isn't a meaningful analog, so the ablation compares
classification metrics only (accuracy/macro-F1/etc.), not total loss.
"""

from __future__ import annotations

import torch
from torch import nn

from fedpda_ids.models.lstm_autoencoder import ClassificationHead


class DNNBaseline(nn.Module):
    """Non-temporal baseline: sees only x[:, -1, :] (the current
    observation), never the preceding 9 timesteps."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden1_units: int = 64,
        latent_dim: int = 32,
        classifier_hidden_units: int = 32,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden1_units)
        self.fc2 = nn.Linear(hidden1_units, latent_dim)
        self.relu = nn.ReLU()
        self.classifier = ClassificationHead(num_classes, latent_dim, classifier_hidden_units, classifier_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 10, F) -- deliberately use ONLY the last timestep
        current = x[:, -1, :]
        h = self.relu(self.fc1(current))
        h = self.relu(self.fc2(h))
        return self.classifier(h)  # logits, (B, num_classes)
