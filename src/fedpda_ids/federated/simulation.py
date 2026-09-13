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
from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import Driver, LegacyContext, ServerApp, SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.simulation import run_simulation

from fedpda_ids.data.sequence_dataset import build_label_index, build_scope_dataloaders, check_client_trainable, get_scope_train_labels
from fedpda_ids.monitoring.metrics_exporter import record_dp_clip_norm, record_round
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
from fedpda_ids.privacy.dp import (
    adaptive_clip_threshold,
    add_gaussian_noise,
    calibrate_noise_multiplier,
    clip_update,
    compute_achieved_epsilon,
    compute_update,
    update_norm,
)

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
    two files ever written, best = lowest centralized val loss).

    `enable_monitoring` (Phase 13, default False): when True, records
    round number + centralized val loss to Prometheus every round.
    Off by default so importing/testing this strategy never binds an
    HTTP port -- callers opt in explicitly via start_metrics_server()
    plus this flag."""

    def __init__(self, *args, checkpoint_dir: Path, run_name: str, run_config: dict, template_model: torch.nn.Module, enable_monitoring: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.run_config = run_config
        self.template_model = template_model
        self.enable_monitoring = enable_monitoring
        self.best_val_loss = float("inf")
        self.best_round = -1
        self.last_val_loss = None

    def evaluate(self, server_round: int, parameters: Parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return result
        loss, metrics = result
        self.last_val_loss = loss
        if self.enable_monitoring:
            record_round(self.run_name, server_round, val_loss=loss)

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

    `enable_monitoring` (Phase 13, default False): see CheckpointingFedAvg's
    docstring -- same opt-in Prometheus recording, off by default.
    """

    def __init__(self, *args, checkpoint_dir: Path, run_name: str, run_config: dict, template_model: torch.nn.Module, enable_monitoring: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.run_config = run_config
        self.template_model = template_model
        self.enable_monitoring = enable_monitoring
        self.best_val_mse = float("inf")
        self.best_round = -1

    def evaluate(self, server_round: int, parameters: Parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return result
        mse, metrics = result  # "loss" IS mse here, see evaluate_fn below
        if self.enable_monitoring:
            record_round(self.run_name, server_round, val_loss=mse)

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


def evaluate_personalized_pool(
    pool: list[int],
    seq_dir: Path,
    client_id_col: str,
    checkpoint_dir: Path,
    run_name: str,
    num_features: int,
    num_classes: int,
    model_cfg: dict,
    window_size: int,
    batch_size: int,
    num_workers: int,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
    lambda_ce: float,
    device: torch.device,
    checkpoint_suffix: str = "best",
) -> tuple[dict, dict]:
    """Per-client final test evaluation, shared by run_personalized_simulation
    and run_dp_personalized_simulation: pairs a shared checkpoint with
    each pool client's own most-recently-saved local head. Clients
    never sampled (no saved head) are skipped and reported, never
    fabricated -- same for clients with genuinely empty test splits.

    `checkpoint_suffix`: which shared checkpoint to evaluate --
    "best" (lowest centralized val MSE, the default, unchanged Phase 6
    behavior) or "last" (final round). PRE-CODING CORRECTION (Phase 8,
    user-approved): under DP, injected noise can make every post-round-0
    val MSE worse than the untrained random init (observed on ALL 16
    real DP sweep runs -- e.g. N-BaIoT eps=8's val MSE: round 0 = 0.141,
    round 1 = 0.320, never recovering over 100 rounds), so "best" always
    picks round 0 -- silently evaluating an UNTRAINED encoder instead of
    the DP-trained one. run_dp_personalized_simulation therefore always
    requests "last" instead; run_personalized_simulation (no DP noise on
    the deployed weights) keeps "best", where this failure mode doesn't
    arise."""
    shared_model = make_model(num_features, num_classes, model_cfg, window_size)
    load_shared_checkpoint(Path(checkpoint_dir) / f"{run_name}_{checkpoint_suffix}.pt", shared_model)

    per_client_test_metrics = {}
    skipped_no_head = []
    for client_id in pool:
        head_path = local_head_path(checkpoint_dir, run_name, client_id)
        client_model = make_model(num_features, num_classes, model_cfg, window_size)
        client_model.load_state_dict(shared_model.state_dict())  # start from the selected shared checkpoint
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
    return per_client_test_metrics, per_client_summary


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
    enable_monitoring: bool = False,
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
        enable_monitoring=enable_monitoring,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(pool),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": max_cpus_per_client, "num_gpus": 0.0},
    )

    per_client_test_metrics, per_client_summary = evaluate_personalized_pool(
        pool=pool, seq_dir=seq_dir, client_id_col=client_id_col, checkpoint_dir=checkpoint_dir, run_name=run_name,
        num_features=num_features, num_classes=num_classes, model_cfg=model_cfg, window_size=window_size,
        batch_size=batch_size, num_workers=num_workers, label_to_index=label_to_index, index_to_label=index_to_label,
        lambda_ce=lambda_ce, device=device,
    )

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


# ---------------------------------------------------------------------
# Phase 8: client-level DP on top of personalized FL's shared encoder/
# decoder updates (see src/fedpda_ids/privacy/dp.py module docstring
# for the full mechanism + the user-approved PRE-CODING CORRECTION on
# uniform-per-client aggregation weighting).
# ---------------------------------------------------------------------


class DPPersonalizedFedAvg(PersonalizedCheckpointingFedAvg):
    """Same shared-parameter FedAvg as PersonalizedCheckpointingFedAvg,
    except aggregate_fit is entirely replaced: instead of a weighted
    average of raw parameters, each client's UPDATE (new shared params
    minus what it started the round with) is clipped to an adaptive
    per-round median threshold, summed, given one shared Gaussian noise
    draw, and divided by a FIXED m=clients_per_round -- never len(results)
    or a client's own dataset size (that would make the aggregation
    weight itself data-dependent, breaking the sensitivity bound the
    accountant's noise calibration relies on). All DP logic lives here,
    entirely server-side -- FlowerLSTMClient itself is unchanged."""

    def __init__(self, *args, noise_multiplier: float, clients_per_round: int, seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.clients_per_round = clients_per_round
        self.rng = np.random.default_rng(seed)
        self.current_global_params: list[np.ndarray] | None = None
        self.clip_norm_history: list[float] = []

    def initialize_parameters(self, client_manager):
        parameters = super().initialize_parameters(client_manager)
        if parameters is not None:
            self.current_global_params = parameters_to_ndarrays(parameters)
        return parameters

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_updates = [
            compute_update(parameters_to_ndarrays(fit_res.parameters), self.current_global_params)
            for _, fit_res in results
        ]
        clip_norm = adaptive_clip_threshold([update_norm(u) for u in client_updates])
        self.clip_norm_history.append(clip_norm)
        if self.enable_monitoring:
            record_dp_clip_norm(self.run_name, clip_norm)

        clipped = [clip_update(u, clip_norm) for u in client_updates]
        summed = clipped[0]
        for u in clipped[1:]:
            summed = [s + a for s, a in zip(summed, u)]

        noised_summed = add_gaussian_noise(summed, self.noise_multiplier, clip_norm, self.rng)
        averaged_update = [arr / self.clients_per_round for arr in noised_summed]
        self.current_global_params = [old + delta for old, delta in zip(self.current_global_params, averaged_update)]

        parameters_aggregated = ndarrays_to_parameters(self.current_global_params)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated["dp_clip_norm"] = clip_norm

        return parameters_aggregated, metrics_aggregated


def run_dp_personalized_simulation(
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
    target_epsilon: float,
    target_delta: float,
    num_workers: int = 0,
    max_cpus_per_client: int = 4,
    enable_monitoring: bool = False,
) -> dict:
    """Same overall shape as run_personalized_simulation, with client-
    level DP applied to the shared encoder/decoder updates (see
    DPPersonalizedFedAvg). target_epsilon=inf means "no DP" (noise_
    multiplier=0.0) -- in practice callers should just reuse an
    existing run_personalized_simulation result for that point of the
    sweep instead of calling this function, but it is handled correctly
    here too (clipping still applies; a real, if less commonly reported,
    ablation).
    """
    seq_dir = Path(seq_dir)

    global_labels = get_scope_train_labels(seq_dir, client_id_col, None)
    label_to_index = build_label_index(global_labels)
    index_to_label = {i: label for label, i in label_to_index.items()}
    num_classes = len(label_to_index)

    pool = build_trainable_client_pool(seq_dir, client_id_col, num_clients_configured, min_train_sequences, min_train_classes)
    logger.info("DP personalized FL client pool: %d/%d configured clients are trainable", len(pool), num_clients_configured)

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
        model = make_model(num_features, num_classes, model_cfg, window_size)
        set_shared_parameters(model, parameters)
        metrics = evaluate(model, val_loader, device, lambda_ce, index_to_label)
        return metrics["mse"], {"val_mse": metrics["mse"]}

    # sample_rate: this round's client-subsampling probability -- the
    # same fraction Flower's fraction_fit/fraction_evaluate use, and
    # what the RDP accountant's calibration assumes.
    sample_rate = clients_per_round / len(pool)
    noise_multiplier = calibrate_noise_multiplier(target_epsilon, target_delta, sample_rate, num_rounds)

    run_config = {
        "run_name": run_name, "personalized": True, "dp": True,
        "target_epsilon": target_epsilon, "target_delta": target_delta,
        "noise_multiplier": noise_multiplier, "sample_rate": sample_rate,
        "num_clients_configured": num_clients_configured, "num_clients_pool": len(pool),
        "clients_per_round": clients_per_round, "num_rounds": num_rounds, "local_epochs": local_epochs,
        "batch_size": batch_size, "seed": seed, "num_features": num_features, "num_classes": num_classes,
        "label_to_index": label_to_index,
    }

    strategy = DPPersonalizedFedAvg(
        fraction_fit=sample_rate, fraction_evaluate=sample_rate,
        min_fit_clients=clients_per_round, min_evaluate_clients=clients_per_round,
        min_available_clients=len(pool),
        evaluate_fn=evaluate_fn, initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        checkpoint_dir=checkpoint_dir, run_name=run_name, run_config=run_config, template_model=template_model,
        noise_multiplier=noise_multiplier, clients_per_round=clients_per_round, seed=seed,
        enable_monitoring=enable_monitoring,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(pool),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": max_cpus_per_client, "num_gpus": 0.0},
    )

    achieved_epsilon = compute_achieved_epsilon(noise_multiplier, sample_rate, num_rounds, target_delta)

    # checkpoint_suffix="last": see evaluate_personalized_pool()'s docstring
    # -- DP noise on the deployed weights can make "best by val MSE" always
    # pick the untrained round-0 checkpoint, so DP runs evaluate the final
    # round instead (real DP-trained weights, never the random init).
    per_client_test_metrics, per_client_summary = evaluate_personalized_pool(
        pool=pool, seq_dir=seq_dir, client_id_col=client_id_col, checkpoint_dir=checkpoint_dir, run_name=run_name,
        num_features=num_features, num_classes=num_classes, model_cfg=model_cfg, window_size=window_size,
        batch_size=batch_size, num_workers=num_workers, label_to_index=label_to_index, index_to_label=index_to_label,
        lambda_ce=lambda_ce, device=device, checkpoint_suffix="last",
    )

    return {
        "history_losses_centralized": history.losses_centralized,
        "history_metrics_centralized": history.metrics_centralized,
        "history_losses_distributed": history.losses_distributed,
        "history_metrics_distributed_fit": history.metrics_distributed_fit,
        "history_metrics_distributed": history.metrics_distributed,
        "best_round": strategy.best_round,
        "best_val_mse": strategy.best_val_mse,
        "clip_norm_history": strategy.clip_norm_history,
        "achieved_epsilon": achieved_epsilon,
        "per_client_evaluation_checkpoint": "last",
        "per_client_test_metrics": per_client_test_metrics,
        "per_client_summary": per_client_summary,
        "run_config": run_config,
        "pool": pool,
        "best_checkpoint": str(checkpoint_dir / f"{run_name}_best.pt"),
        "last_checkpoint": str(checkpoint_dir / f"{run_name}_last.pt"),
    }


# ---------------------------------------------------------------------
# Phase 9: Flower SecAgg+ on top of PLAIN (non-DP) personalized FL.
#
# SecAgg+ cryptographically sums client updates so the server never sees
# any individual client's update -- only the (masked, secret-shared) sum.
# This is fundamentally incompatible with Phase 8's ADAPTIVE clipping
# (median of individual update norms requires seeing individual values),
# so per user decision, SecAgg+ is evaluated as its own independent
# mechanism on the plain (no-DP) personalized pipeline, never combined
# with Phase 8's DP mechanism -- "DP and SecAgg are different
# mechanisms" per the frozen spec.
#
# SecAgg+ requires Flower's newer ServerApp/ClientApp/run_simulation
# harness, not the legacy start_simulation(strategy=...) API Phases
# 5/6/8 use -- but LegacyContext lets our EXISTING, UNCHANGED
# PersonalizedCheckpointingFedAvg strategy plug into it directly:
# SecAgg+ handles secure summation of whatever parameters clients
# return (the shared encoder/decoder, exactly as before), then hands
# the revealed (correctly-averaged) result to the same strategy.evaluate()
# override for checkpointing -- no new Strategy subclass needed.
#
# Two real, pre-validated (standalone smoke test) parameter choices:
# - clipping_range=16.0: SecAgg+ quantizes real-valued params into
#   bounded integers for secret-sharing; values outside this range are
#   silently clipped (real corruption, not an error). Measured real
#   trained encoder/decoder weight magnitude on this project's data:
#   max |weight| = 10.19 -- clipping_range=16.0 gives ~1.6x headroom.
# - modulus_range=2**30: Flower's default (2**32) overflows
#   numpy.random.RandomState.randint's int32 limit on this platform
#   (confirmed via standalone smoke test -- a genuine Flower/numpy
#   compatibility bug, not something specific to our parameters/data).
#   2**30 is comfortably under int32's limit while still far exceeding
#   the actual required capacity (num_clients * max_weight_ratio *
#   quantization_range, at most a few hundred million for this project's
#   scopes) -- see max_weight below for why the ratio stays small.
# - max_weight=50000.0: SecAgg+ uses ratio = num_examples / max_weight
#   as each client's actual weighting factor (NOT num_examples itself),
#   so max_weight must exceed every client's real train-sequence count
#   (measured max: 12,640 CICIDS2017, 28,485 N-BaIoT) or that client's
#   weight silently overflows the protocol's capacity.
# ---------------------------------------------------------------------


def run_secagg_personalized_simulation(
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
    clipping_range: float = 16.0,
    max_weight: float = 50000.0,
    modulus_range: int = 2**30,
) -> dict:
    """Same overall shape as run_personalized_simulation, with SecAgg+
    securely summing the shared encoder/decoder updates so the server
    never observes any individual client's contribution -- only the
    final (correctly-averaged) result, exactly as plain FedAvg would
    have produced (verified via a standalone smoke test against a
    manually-computed expected weighted average)."""
    seq_dir = Path(seq_dir)

    global_labels = get_scope_train_labels(seq_dir, client_id_col, None)
    label_to_index = build_label_index(global_labels)
    index_to_label = {i: label for label, i in label_to_index.items()}
    num_classes = len(label_to_index)

    pool = build_trainable_client_pool(seq_dir, client_id_col, num_clients_configured, min_train_sequences, min_train_classes)
    logger.info("SecAgg+ personalized FL client pool: %d/%d configured clients are trainable", len(pool), num_clients_configured)

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

    client_app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])

    fraction = clients_per_round / len(pool)
    run_config = {
        "run_name": run_name, "personalized": True, "secagg_plus": True,
        "num_clients_configured": num_clients_configured, "num_clients_pool": len(pool),
        "clients_per_round": clients_per_round, "num_rounds": num_rounds, "local_epochs": local_epochs,
        "batch_size": batch_size, "seed": seed, "num_features": num_features, "num_classes": num_classes,
        "label_to_index": label_to_index, "clipping_range": clipping_range, "max_weight": max_weight,
        "modulus_range": modulus_range,
    }

    captured: dict = {}

    server_app = ServerApp()

    @server_app.main()
    def main(driver: Driver, context: Context) -> None:  # noqa: ANN001
        template_model = make_model(num_features, num_classes, model_cfg, window_size)
        initial_parameters = ndarrays_to_parameters(get_shared_parameters(template_model))

        def evaluate_fn(server_round: int, parameters, config):
            model = make_model(num_features, num_classes, model_cfg, window_size)
            set_shared_parameters(model, parameters)
            metrics = evaluate(model, val_loader, device, lambda_ce, index_to_label)
            return metrics["mse"], {"val_mse": metrics["mse"]}

        strategy = PersonalizedCheckpointingFedAvg(
            fraction_fit=fraction, fraction_evaluate=fraction,
            min_fit_clients=clients_per_round, min_evaluate_clients=clients_per_round,
            min_available_clients=len(pool),
            evaluate_fn=evaluate_fn, initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
            checkpoint_dir=checkpoint_dir, run_name=run_name, run_config=run_config, template_model=template_model,
        )

        legacy_context = LegacyContext(
            context=context,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_manager=SimpleClientManager(),
        )
        # num_shares/reconstruction_threshold size the secret-sharing scheme
        # among the clients ACTUALLY SAMPLED in a given round (clients_per_round),
        # not the full trainable pool -- a real bug found via a real-data smoke
        # test (CICIDS2017 alpha=5: pool=40, clients_per_round=8): using
        # len(pool) here made every round's SecAgg+ handshake wait for shares
        # from clients that were never even asked to participate that round,
        # so no client ever got a personalized head trained. The tiny
        # synthetic tests never caught this because they used
        # num_clients_configured == clients_per_round (every client sampled
        # every round), coincidentally making the two values equal.
        fit_workflow = SecAggPlusWorkflow(
            num_shares=clients_per_round,
            reconstruction_threshold=max(1, clients_per_round - 1),
            max_weight=max_weight,
            clipping_range=clipping_range,
            modulus_range=modulus_range,
        )
        workflow = DefaultWorkflow(fit_workflow=fit_workflow)
        workflow(driver, legacy_context)

        captured["history"] = legacy_context.history
        captured["best_round"] = strategy.best_round
        captured["best_val_mse"] = strategy.best_val_mse

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(pool),
        backend_config={"client_resources": {"num_cpus": max_cpus_per_client, "num_gpus": 0.0}},
    )

    history = captured["history"]

    per_client_test_metrics, per_client_summary = evaluate_personalized_pool(
        pool=pool, seq_dir=seq_dir, client_id_col=client_id_col, checkpoint_dir=checkpoint_dir, run_name=run_name,
        num_features=num_features, num_classes=num_classes, model_cfg=model_cfg, window_size=window_size,
        batch_size=batch_size, num_workers=num_workers, label_to_index=label_to_index, index_to_label=index_to_label,
        lambda_ce=lambda_ce, device=device, checkpoint_suffix="best",
    )

    return {
        "history_losses_centralized": history.losses_centralized,
        "history_metrics_centralized": history.metrics_centralized,
        "history_losses_distributed": history.losses_distributed,
        "history_metrics_distributed_fit": history.metrics_distributed_fit,
        "history_metrics_distributed": history.metrics_distributed,
        "best_round": captured["best_round"],
        "best_val_mse": captured["best_val_mse"],
        "per_client_evaluation_checkpoint": "best",
        "per_client_test_metrics": per_client_test_metrics,
        "per_client_summary": per_client_summary,
        "run_config": run_config,
        "pool": pool,
        "best_checkpoint": str(checkpoint_dir / f"{run_name}_best.pt"),
        "last_checkpoint": str(checkpoint_dir / f"{run_name}_last.pt"),
    }
