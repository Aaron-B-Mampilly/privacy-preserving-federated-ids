"""Phase 4: the frozen LSTM autoencoder + classification head architecture.

Shared encoder/decoder (will become the federated component in later
phases) plus a local classification head (stays client-side per the
frozen spec: "C_k can differ between clients because the
classification head is personalized/local").

Shapes, exactly per the frozen spec:
    input:      (B, 10, F)
    encoder ->  (B, 32)              latent
    decoder ->  (B, 10, F)           reconstruction
    classifier -> (B, C)             logits (no softmax -- CrossEntropyLoss applies it internally)
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, num_features: int, layer1_units: int = 64, latent_dim: int = 32):
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, layer1_units, batch_first=True)
        self.lstm2 = nn.LSTM(layer1_units, latent_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 10, F)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return out[:, -1, :]  # (B, latent_dim) -- final timestep's hidden state


class LSTMDecoder(nn.Module):
    def __init__(self, num_features: int, window_size: int = 10, latent_dim: int = 32, layer1_units: int = 64):
        super().__init__()
        self.window_size = window_size
        self.lstm1 = nn.LSTM(latent_dim, layer1_units, batch_first=True)
        self.lstm2 = nn.LSTM(layer1_units, num_features, batch_first=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (B, latent_dim) -- RepeatVector(window_size) equivalent
        repeated = latent.unsqueeze(1).repeat(1, self.window_size, 1)  # (B, W, latent_dim)
        out, _ = self.lstm1(repeated)
        out, _ = self.lstm2(out)
        return out  # (B, W, F)


class ClassificationHead(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 32, hidden_units: int = 32, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(latent))
        x = self.dropout(x)
        return self.fc2(x)  # logits, (B, num_classes)


class LSTMAutoencoderClassifier(nn.Module):
    """The full frozen architecture: shared encoder/decoder + local
    classification head, wired together for single-process training.
    (Federated separation of encoder/decoder vs. classifier is a
    Phase 5/6 concern -- this module just needs the forward math right.)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        window_size: int = 10,
        latent_dim: int = 32,
        encoder_layer1_units: int = 64,
        decoder_layer1_units: int = 64,
        classifier_hidden_units: int = 32,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.encoder = LSTMEncoder(num_features, encoder_layer1_units, latent_dim)
        self.decoder = LSTMDecoder(num_features, window_size, latent_dim, decoder_layer1_units)
        self.classifier = ClassificationHead(num_classes, latent_dim, classifier_hidden_units, classifier_dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        logits = self.classifier(latent)
        return reconstruction, logits, latent


def compute_total_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    lambda_ce: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """total_loss = MSE_recon + lambda_ce * CE. Returns (total, mse, ce)
    so callers can log the two components separately."""
    mse = nn.functional.mse_loss(reconstruction, x)
    ce = nn.functional.cross_entropy(logits, y)
    total = mse + lambda_ce * ce
    return total, mse, ce
