"""Phase 5: Flower FedAvg simulation orchestration.

Design recap (see client.py docstring for the full reasoning):
- ONE global class vocabulary shared by every client (required for
  FedAvg to be valid across differently-labeled clients).
- The client pool is filtered through Phase 4's check_client_trainable
  policy (>=10 train sequences, >=2 classes) -- a client that fails it
  is excluded from the simulation entirely, not fabricated a viable
  update. Flower's `num_clients` is set to the SIZE OF THE FILTERED
  POOL, and client_fn maps Flower's partition-id (0..num_clients-1,
  confirmed via direct probe of this Flower version's Context API --
  NOT the same as its internal node_id) to that pool's actual client
  ids.
- Server-side evaluation each round uses the CENTRALIZED VALIDATION
  set (never test -- same "don't tune on test" rule as Phase 4);
  test is evaluated exactly once, after the simulation, using the
  best-by-val-loss round's weights.
- Client-side evaluate() (distributed, on each sampled client's own
  val split) is also wired up and its Flower-aggregated metrics are
  kept in the returned History, but the centralized val/test
  evaluation is what best_checkpoint selection and the final report
  are based on -- consistent with Phase 4's methodology.
"""

from __future__ import annotations

import logging
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fedpda_ids.data.sequence_dataset import build_label_index, build_scope_dataloaders, check_client_trainable, get_scope_train_labels
from fedpda_ids.federated.client import (
    FlowerLSTMClient,
    get_model_parameters,
    get_shared_parameters,
    load_local_head,
    local_head_path,
    set_model_parameters,
    set_shared_parameters,
)
from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier
from fedpda_ids.models.trainer import evaluate, save_checkpoint

logger = logging.getLogger("fedpda_ids")


def build_trainable_client_pool(
    seq_dir: Path, client_id_col: str, num_clients_configured: int,
    min_train_sequences: int, min_train_classes: int,
) -> list[int]:
    """The ordered list of client ids (0..num_clients_configured-1) that
    clear Phase 4's trainability bar. Flower's partition-id indexes
    INTO this list, not into the raw client id space directly, so a
    filtered-out client (e.g. CICIDS2017 alpha=0.1's client 18) never
    gets sampled at all."""
    pool = []
    for cid in range(num_clients_configured):
        ok, reason = check_client_trainable(seq_dir, client_id_col, cid, min_train_sequences, min_train_classes)
        if ok:
            pool.append(cid)
        else:
            logger.warning("Client %d excluded from FL pool: %s", cid, reason)
    return pool


def make_model(num_features: int, num_classes: int, model_cfg: dict, window_size: int) -> torch.nn.Module:
    return LSTMAutoencoderClassifier(
        num_features=num_features,
        num_classes=num_classes,
        window_size=window_size,
        latent_dim=model_cfg["latent_dim"],
        encoder_layer1_units=model_cfg["encoder"]["layer1_units"],
        decoder_layer1_units=model_cfg["decoder"]["layer1_units"],
        classifier_hidden_units=model_cfg["classifier_head"]["hidden_units"],
        classifier_dropout=model_cfg["classifier_head"]["dropout"],
    )


class CheckpointingFedAvg(FedAvg):
    """Plain flwr.server.strategy.FedAvg, plus saving best/last global
    checkpoints -- mirrors Phase 4's train_model() convention (only
    two files ever written, best = lowest centralized val loss)."""

    def __init__(self, *args, checkpoint_dir: Path, run_name: str, run_config: dict, template_model: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.run_config = run_config
        self.template_model = template_model
        self.best_val_loss = float("inf")
        self.best_round = -1
        self.last_val_loss = None

    def evaluate(self, server_round: int, parameters: Parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return result
        loss, metrics = result
        self.last_val_loss = loss

        set_model_parameters(self.template_model, parameters_to_ndarrays(parameters))
        dummy_optimizer = torch.optim.Adam(self.template_model.parameters(), lr=1e-3)

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_round = server_round
            save_checkpoint(
                self.checkpoint_dir / f"{self.run_name}_best.pt", self.template_model, dummy_optimizer,
                server_round, self.run_config, {}, {"loss": loss, **metrics},
            )
        save_checkpoint(
            self.checkpoint_dir / f"{self.run_name}_last.pt", self.template_model, dummy_optimizer,
            server_round, self.run_config, {}, {"loss": loss, **metrics},
        )
        return result


def run_fedavg_simulation(
    seq_dir: Path,
    client_id_col: str,
    num_clients_configured: int,
    clients_per_round: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    model_cfg: dict,
    window_size: int,
    min_train_sequences: int,
    min_train_classes: int,
    checkpoint_dir: Path,
    run_name: str,
    seed: int,
    num_workers: int = 0,
    max_cpus_per_client: int = 4,
) -> dict:
    seq_dir = Path(seq_dir)

    # Global class vocabulary: same policy as Phase 4's centralized
    # scope (client_id=None means "union of all clients' train data").
    global_labels = get_scope_train_labels(seq_dir, client_id_col, None)
    label_to_index = build_label_index(global_labels)
    index_to_label = {i: label for label, i in label_to_index.items()}
    num_classes = len(label_to_index)

    pool = build_trainable_client_pool(seq_dir, client_id_col, num_clients_configured, min_train_sequences, min_train_classes)
    logger.info("FL client pool: %d/%d configured clients are trainable", len(pool), num_clients_configured)

    device = torch.device("cpu")  # simulation runs many concurrent virtual clients; see module docstring
    lambda_ce = model_cfg["loss"]["lambda_ce"]
    learning_rate = model_cfg["optimizer"]["learning_rate"]

    # Peek num_features from the centralized scope's feature count.
    centralized_scope = build_scope_dataloaders(seq_dir, client_id_col, None, batch_size, num_workers, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]
    val_loader = centralized_scope["loaders"]["val"]
    test_loader = centralized_scope["loaders"]["test"]

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        client_id = pool[partition_id]

        scope = build_scope_dataloaders(
            seq_dir, client_id_col, client_id, batch_size, num_workers, False, label_to_index,
        )
        model = make_model(num_features, num_classes, model_cfg, window_size)
        client = FlowerLSTMClient(
            client_id=client_id, model=model,
            train_loader=scope["loaders"]["train"], val_loader=scope["loaders"]["val"],
            device=device, local_epochs=local_epochs, lambda_ce=lambda_ce,
            learning_rate=learning_rate, index_to_label=index_to_label,
        )
        return client.to_client()

    template_model = make_model(num_features, num_classes, model_cfg, window_size)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(template_model))

    def evaluate_fn(server_round: int, parameters, config):
        model = make_model(num_features, num_classes, model_cfg, window_size)
        set_model_parameters(model, parameters)
        metrics = evaluate(model, val_loader, device, lambda_ce, index_to_label)
        return metrics["loss"], {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    fraction = clients_per_round / len(pool)
    run_config = {
        "run_name": run_name, "num_clients_configured": num_clients_configured, "num_clients_pool": len(pool),
        "clients_per_round": clients_per_round, "num_rounds": num_rounds, "local_epochs": local_epochs,
        "batch_size": batch_size, "seed": seed, "num_features": num_features, "num_classes": num_classes,
        "label_to_index": label_to_index,
    }

    strategy = CheckpointingFedAvg(
        fraction_fit=fraction, fraction_evaluate=fraction,
        min_fit_clients=clients_per_round, min_evaluate_clients=clients_per_round,
        min_available_clients=len(pool),
        evaluate_fn=evaluate_fn, initial_parameters=initial_parameters,
        checkpoint_dir=checkpoint_dir, run_name=run_name, run_config=run_config, template_model=template_model,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(pool),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        # num_cpus=1 per client lets Ray run one actor per logical CPU
        # (16 here) -- each independently imports this venv's CUDA-
        # enabled torch, which eagerly loads several hundred MB of CUDA
        # runtime DLLs on import regardless of whether GPU is used.
        # 16 concurrent copies of that exhausted this machine's paging
        # file mid-run (a real client task failure, not hypothetical --
        # see Phase 5 writeup). Asking for more CPUs per client forces
        # Ray to run fewer actors concurrently, capping how many
        # torch-loaded processes exist at once. Phase 4 already found
        # GPU gives this small model no speedup, so this costs nothing.
        client_resources={"num_cpus": max_cpus_per_client, "num_gpus": 0.0},
    )

    # Final test evaluation, exactly once, using the best-val-loss round's weights.
    best_ckpt = torch.load(Path(checkpoint_dir) / f"{run_name}_best.pt", map_location="cpu", weights_only=False)
    test_model = make_model(num_features, num_classes, model_cfg, window_size)
    test_model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(test_model, test_loader, device, lambda_ce, index_to_label)

    return {
        "history_losses_centralized": history.losses_centralized,
        "history_metrics_centralized": history.metrics_centralized,
        "history_losses_distributed": history.losses_distributed,
        "history_metrics_distributed_fit": history.metrics_distributed_fit,
        "history_metrics_distributed": history.metrics_distributed,
        "best_round": strategy.best_round,
        "best_val_loss": strategy.best_val_loss,
        "test_metrics": test_metrics,
        "run_config": run_config,
        "pool": pool,
        "best_checkpoint": str(Path(checkpoint_dir) / f"{run_name}_best.pt"),
        "last_checkpoint": str(Path(checkpoint_dir) / f"{run_name}_last.pt"),
    }


# ---------------------------------------------------------------------
# Phase 6: personalized FL (shared encoder/decoder, local classifier head)
# ---------------------------------------------------------------------


def weighted_average_metrics(results: list[tuple[int, dict]]) -> dict:
    """Flower metrics-aggregation callback: weights each client's
    metrics by its num_examples. Used for the DISTRIBUTED classification
    metrics, which are the primary classification-quality signal under
    personalization (there is no single global classifier to evaluate
    centrally -- see module docstring)."""
    total_examples = sum(n for n, _ in results)
    if total_examples == 0 or not results:
        return {}
    keys = results[0][1].keys()
    return {k: sum(n * m[k] for n, m in results) / total_examples for k in keys}


def save_shared_checkpoint(path: Path, model: torch.nn.Module, round_num: int, run_config: dict, metrics: dict) -> None:
    """Persists ONLY the shared encoder/decoder tensors + metadata --
    deliberately NOT the full model.state_dict(), so this file can
    never be mistaken for containing a meaningful trained classifier
    (there isn't one at the server under personalization)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "shared_state_dict": {k: v for k, v in model.state_dict().items() if k.startswith(("encoder.", "decoder."))},
            "round": round_num,
            "run_config": run_config,
            "metrics": metrics,
        },
        path,
    )


def load_shared_checkpoint(path: Path, model: torch.nn.Module) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["shared_state_dict"], strict=False)
    return ckpt


class PersonalizedCheckpointingFedAvg(FedAvg):
    """FedAvg restricted to shared (encoder/decoder) parameters --
    Flower's aggregation itself is shape-agnostic, so no override is
    needed there; what changes is the CENTRALIZED evaluation, which
    can only meaningfully assess the shared encoder/decoder via
    reconstruction MSE (there is no global classifier). Best round =
    lowest centralized val MSE -- a deliberately new, explicitly
    documented criterion for this phase's different model structure,
    NOT a retroactive change to Phase 5's "lowest total loss" policy.
    """

    def __init__(self, *args, checkpoint_dir: Path, run_name: str, run_config: dict, template_model: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.run_config = run_config
        self.template_model = template_model
        self.best_val_mse = float("inf")
        self.best_round = -1

    def evaluate(self, server_round: int, parameters: Parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return result
        mse, metrics = result  # "loss" IS mse here, see evaluate_fn below

        set_shared_parameters(self.template_model, parameters_to_ndarrays(parameters))

        if mse < self.best_val_mse:
            self.best_val_mse = mse
            self.best_round = server_round
            save_shared_checkpoint(
                self.checkpoint_dir / f"{self.run_name}_best.pt", self.template_model,
                server_round, self.run_config, {"val_mse": mse, **metrics},
            )
        save_shared_checkpoint(
            self.checkpoint_dir / f"{self.run_name}_last.pt", self.template_model,
            server_round, self.run_config, {"val_mse": mse, **metrics},
        )
        return result


def run_personalized_simulation(
    seq_dir: Path,
    client_id_col: str,
    num_clients_configured: int,
    clients_per_round: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    model_cfg: dict,
    window_size: int,
    min_train_sequences: int,
    min_train_classes: int,
    checkpoint_dir: Path,
    run_name: str,
    seed: int,
    num_workers: int = 0,
    max_cpus_per_client: int = 4,
) -> dict:
    """Phase 6: same overall shape as run_fedavg_simulation, but only
    encoder/decoder parameters are exchanged; each client keeps its
    own classification head (disk-persisted -- see client.py). Final
    evaluation is PER-CLIENT (there's no single global classifier to
    report one test number for), pairing the best shared checkpoint
    with each client's own most-recently-saved local head.
    """
    seq_dir = Path(seq_dir)

    global_labels = get_scope_train_labels(seq_dir, client_id_col, None)
    label_to_index = build_label_index(global_labels)
    index_to_label = {i: label for label, i in label_to_index.items()}
    num_classes = len(label_to_index)

    pool = build_trainable_client_pool(seq_dir, client_id_col, num_clients_configured, min_train_sequences, min_train_classes)
    logger.info("Personalized FL client pool: %d/%d configured clients are trainable", len(pool), num_clients_configured)

    device = torch.device("cpu")
    lambda_ce = model_cfg["loss"]["lambda_ce"]
    learning_rate = model_cfg["optimizer"]["learning_rate"]

    centralized_scope = build_scope_dataloaders(seq_dir, client_id_col, None, batch_size, num_workers, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]
    val_loader = centralized_scope["loaders"]["val"]

    checkpoint_dir = Path(checkpoint_dir)

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        client_id = pool[partition_id]

        scope = build_scope_dataloaders(
            seq_dir, client_id_col, client_id, batch_size, num_workers, False, label_to_index,
        )
        model = make_model(num_features, num_classes, model_cfg, window_size)
        client = FlowerLSTMClient(
            client_id=client_id, model=model,
            train_loader=scope["loaders"]["train"], val_loader=scope["loaders"]["val"],
            device=device, local_epochs=local_epochs, lambda_ce=lambda_ce,
            learning_rate=learning_rate, index_to_label=index_to_label,
            personalized=True, head_path=local_head_path(checkpoint_dir, run_name, client_id),
        )
        return client.to_client()

    template_model = make_model(num_features, num_classes, model_cfg, window_size)
    initial_parameters = ndarrays_to_parameters(get_shared_parameters(template_model))

    def evaluate_fn(server_round: int, parameters, config):
        # Reconstruction MSE only -- the template's classifier is never
        # trained under personalization, so its logits/CE are meaningless
        # and must not be reported as if they were a real metric.
        model = make_model(num_features, num_classes, model_cfg, window_size)
        set_shared_parameters(model, parameters)
        metrics = evaluate(model, val_loader, device, lambda_ce, index_to_label)
        return metrics["mse"], {"val_mse": metrics["mse"]}

    fraction = clients_per_round / len(pool)
    run_config = {
        "run_name": run_name, "personalized": True,
        "num_clients_configured": num_clients_configured, "num_clients_pool": len(pool),
        "clients_per_round": clients_per_round, "num_rounds": num_rounds, "local_epochs": local_epochs,
        "batch_size": batch_size, "seed": seed, "num_features": num_features, "num_classes": num_classes,
        "label_to_index": label_to_index,
    }

    strategy = PersonalizedCheckpointingFedAvg(
        fraction_fit=fraction, fraction_evaluate=fraction,
        min_fit_clients=clients_per_round, min_evaluate_clients=clients_per_round,
        min_available_clients=len(pool),
        evaluate_fn=evaluate_fn, initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        checkpoint_dir=checkpoint_dir, run_name=run_name, run_config=run_config, template_model=template_model,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(pool),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": max_cpus_per_client, "num_gpus": 0.0},
    )

    # Per-client final test evaluation: best shared checkpoint + each
    # client's own most-recently-saved local head. Clients never
    # sampled (no saved head) are skipped and reported, not fabricated.
    best_shared_model = make_model(num_features, num_classes, model_cfg, window_size)
    load_shared_checkpoint(checkpoint_dir / f"{run_name}_best.pt", best_shared_model)

    per_client_test_metrics = {}
    skipped_no_head = []
    for client_id in pool:
        head_path = local_head_path(checkpoint_dir, run_name, client_id)
        client_model = make_model(num_features, num_classes, model_cfg, window_size)
        client_model.load_state_dict(best_shared_model.state_dict())  # start from best shared weights
        found = load_local_head(client_model, head_path)
        if not found:
            skipped_no_head.append(client_id)
            continue

        client_scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, batch_size, num_workers, False, label_to_index)
        test_loader = client_scope["loaders"]["test"]
        if len(client_scope["datasets"]["test"]) == 0:
            per_client_test_metrics[client_id] = {"status": "empty_test_split"}
            continue
        metrics = evaluate(client_model, test_loader, device, lambda_ce, index_to_label)
        per_client_test_metrics[client_id] = metrics

    if skipped_no_head:
        logger.warning("%d/%d pool clients were never sampled during training (no personalized head learned): %s",
                        len(skipped_no_head), len(pool), skipped_no_head)

    valid_metrics = [m for m in per_client_test_metrics.values() if "status" not in m]
    accs = [m["accuracy"] for m in valid_metrics]
    macro_f1s = [m["macro_f1"] for m in valid_metrics]
    per_client_summary = {
        "num_clients_evaluated": len(valid_metrics),
        "num_clients_skipped_no_head": len(skipped_no_head),
        "skipped_client_ids": skipped_no_head,
        "accuracy_mean": float(np.mean(accs)) if accs else None,
        "accuracy_std": float(np.std(accs)) if accs else None,
        "macro_f1_mean": float(np.mean(macro_f1s)) if macro_f1s else None,
        "macro_f1_std": float(np.std(macro_f1s)) if macro_f1s else None,
    }

    return {
        "history_losses_centralized": history.losses_centralized,
        "history_metrics_centralized": history.metrics_centralized,
        "history_losses_distributed": history.losses_distributed,
        "history_metrics_distributed_fit": history.metrics_distributed_fit,
        "history_metrics_distributed": history.metrics_distributed,
        "best_round": strategy.best_round,
        "best_val_mse": strategy.best_val_mse,
        "per_client_test_metrics": per_client_test_metrics,
        "per_client_summary": per_client_summary,
        "run_config": run_config,
        "pool": pool,
        "best_checkpoint": str(checkpoint_dir / f"{run_name}_best.pt"),
        "last_checkpoint": str(checkpoint_dir / f"{run_name}_last.pt"),
    }
