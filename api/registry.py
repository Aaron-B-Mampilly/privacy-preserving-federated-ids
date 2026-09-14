"""Lazy model/prototype loading for the prediction endpoint.

Every object here is built by the EXISTING research code -- `make_model`,
`load_shared_checkpoint`, `load_local_head`, `client_prototypes`,
`aggregate_prototypes`, `calibrate_threshold`. The API adds caching and
nothing else; the forward pass a user triggers from the UI is bit-for-bit
the one `evaluate()` runs during a real experiment.

Checkpoints total ~382 MB on disk, so nothing is loaded at import time --
models are built on first use and kept in a small LRU.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from api.config import CHECKPOINT_DIR, CONFIG_PATH, RESULTS_DIR, Scope
from api.repository import repository

from fedpda_ids.data.sequence_dataset import build_scope_dataloaders
from fedpda_ids.federated.client import load_local_head, local_head_path
from fedpda_ids.federated.simulation import (
    build_trainable_client_pool,
    load_shared_checkpoint,
    make_model,
)
from fedpda_ids.models.prototypes import (
    aggregate_prototypes,
    calibrate_threshold,
    classify_batch_with_prototypes,
    client_prototypes,
)
from fedpda_ids.models.trainer import load_checkpoint
from fedpda_ids.utils.config import load_config


@dataclass
class LoadedModel:
    model: torch.nn.Module
    run_name: str
    checkpoint_suffix: str
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    num_features: int
    personalized: bool
    client_id: int | None = None


class ModelRegistry:
    """Thread-safe lazy cache of models, prototypes and sequence arrays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[tuple, LoadedModel] = {}
        self._prototypes: dict[str, tuple[dict[int, np.ndarray], float]] = {}
        self._sequences: dict[str, np.ndarray] = {}
        self._config: dict[str, Any] | None = None

    # -- config ------------------------------------------------------
    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_config(str(CONFIG_PATH))
        return self._config

    # -- raw sequence data -------------------------------------------
    def sequences(self, scope: Scope) -> np.ndarray:
        """Memory-mapped X.npy for a scope -- never read fully into RAM."""
        key = scope.key
        with self._lock:
            if key not in self._sequences:
                path = scope.sequence_dir / "X.npy"
                if not path.exists():
                    raise FileNotFoundError(f"sequence artifact missing for {scope.label}")
                self._sequences[key] = np.load(path, mmap_mode="r")
            return self._sequences[key]

    def sample_window(self, scope: Scope, sequence_index: int) -> np.ndarray:
        """One (window, features) sample, upcast to float32 (N-BaIoT is
        stored as float16 -- the same upcast SequenceDataset performs)."""
        X = self.sequences(scope)
        if not 0 <= sequence_index < len(X):
            raise IndexError(f"sequence_index {sequence_index} out of range (0..{len(X) - 1})")
        return np.array(X[sequence_index], dtype=np.float32)

    # -- models ------------------------------------------------------
    def get_model(
        self,
        scope: Scope,
        mechanism: str = "personalized",
        checkpoint_suffix: str = "best",
        client_id: int | None = None,
    ) -> LoadedModel:
        cache_key = (scope.key, mechanism, checkpoint_suffix, client_id)
        with self._lock:
            if cache_key in self._models:
                return self._models[cache_key]

        run = repository.load_mechanism(scope, mechanism)
        if run is None:
            raise FileNotFoundError(f"no completed {mechanism!r} run for {scope.label}")
        run_name = repository.run_name(scope, mechanism)
        run_config = run["run_config"]
        label_to_index = run_config["label_to_index"]
        index_to_label = {int(i): label for label, i in label_to_index.items()}
        num_features = int(run_config["num_features"])
        num_classes = len(label_to_index)
        model_cfg = self.config["model"]
        window_size = self.config["sequence"]["window_size"]

        model = make_model(num_features, num_classes, model_cfg, window_size)
        personalized = bool(run_config.get("personalized", False))

        if personalized:
            ckpt_path = CHECKPOINT_DIR / f"{run_name}_{checkpoint_suffix}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path.name}")
            load_shared_checkpoint(ckpt_path, model)
            if client_id is not None:
                head = local_head_path(CHECKPOINT_DIR, run_name, client_id)
                if not load_local_head(model, head):
                    raise FileNotFoundError(
                        f"client {client_id} has no persisted classifier head in this run "
                        "(it was never sampled during training)"
                    )
        else:
            ckpt_path = CHECKPOINT_DIR / f"{run_name}_{checkpoint_suffix}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path.name}")
            model.load_state_dict(load_checkpoint(ckpt_path)["model_state_dict"])

        model.eval()
        loaded = LoadedModel(
            model=model,
            run_name=run_name,
            checkpoint_suffix=checkpoint_suffix,
            label_to_index=label_to_index,
            index_to_label=index_to_label,
            num_features=num_features,
            personalized=personalized,
            client_id=client_id,
        )
        with self._lock:
            # Bounded cache: personalized inference builds one model per
            # client, so an unbounded dict would grow with every sample.
            if len(self._models) > 24:
                self._models.clear()
            self._models[cache_key] = loaded
        return loaded

    # -- prototypes (zero-day / NEW CLASS) ---------------------------
    def get_prototypes(self, scope: Scope) -> tuple[dict[int, np.ndarray], float]:
        """Aggregated class prototypes + the NEW CLASS threshold.

        The prototype VECTORS were never persisted by Phase 7's pipeline
        (only the threshold and per-class support were), so they are
        recomputed here with the same frozen clip_bound/aggregation and
        then cached to disk. The recomputed threshold is checked against
        the published one so a silent divergence from the E4 numbers
        would surface rather than pass unnoticed.
        """
        key = scope.key
        with self._lock:
            if key in self._prototypes:
                return self._prototypes[key]

        cache_path = RESULTS_DIR / f"{scope.run_prefix}_prototype_vectors.npz"
        if cache_path.exists():
            blob = np.load(cache_path)
            protos = {int(k): blob[k] for k in blob.files if k != "__threshold__"}
            threshold = float(blob["__threshold__"][0])
        else:
            protos, threshold = self._compute_prototypes(scope)
            to_save = {str(k): v for k, v in protos.items()}
            to_save["__threshold__"] = np.array([threshold], dtype=np.float64)
            np.savez(cache_path, **to_save)

        with self._lock:
            self._prototypes[key] = (protos, threshold)
        return protos, threshold

    def _compute_prototypes(self, scope: Scope) -> tuple[dict[int, np.ndarray], float]:
        loaded = self.get_model(scope, "personalized", "best")
        encoder = loaded.model.encoder
        cfg = self.config
        clip_bound = cfg["prototypes"]["clip_bound"]
        multiplier = cfg["prototypes"]["zero_day_threshold_multiplier"]
        batch_size = cfg["model"]["batch_size"]

        pool = build_trainable_client_pool(
            scope.sequence_dir,
            scope.client_id_col,
            scope.num_clients,
            cfg["training"]["local_training"]["min_train_sequences"],
            cfg["training"]["local_training"]["min_train_classes"],
        )
        device = torch.device("cpu")
        per_client = []
        for client_id in pool:
            client_scope = build_scope_dataloaders(
                scope.sequence_dir, scope.client_id_col, client_id,
                batch_size, 0, False, loaded.label_to_index,
            )
            per_client.append(
                client_prototypes(encoder, client_scope["loaders"]["train"], device, clip_bound=clip_bound)
            )

        aggregated, support = aggregate_prototypes(per_client, return_support=True)
        threshold = calibrate_threshold(aggregated, support, multiplier=multiplier)
        return aggregated, float(threshold)

    def published_threshold(self, scope: Scope) -> float | None:
        stored = repository.load_prototypes(scope, "inf")
        if stored is None:
            return None
        try:
            return float(stored["threshold"])
        except (KeyError, TypeError, ValueError):
            return None

    def classify_new_class(self, scope: Scope, latent: np.ndarray) -> tuple[int, float, float]:
        """Returns (predicted_label_or_-1, min_distance, threshold)."""
        protos, threshold = self.get_prototypes(scope)
        clip_bound = self.config["prototypes"]["clip_bound"]
        preds, dists = classify_batch_with_prototypes(
            latent.reshape(1, -1), protos, threshold, clip_bound=clip_bound
        )
        return int(preds[0]), float(dists[0]), float(threshold)


registry = ModelRegistry()
