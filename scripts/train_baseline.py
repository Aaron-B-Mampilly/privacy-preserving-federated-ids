"""Phase 4: centralized/local LSTM autoencoder+classifier baseline.

Usage:
    # centralized, CICIDS2017 alpha=5
    python scripts/train_baseline.py --dataset cicids2017 --alpha 5 --scope centralized

    # local-only, one CICIDS2017 client
    python scripts/train_baseline.py --dataset cicids2017 --alpha 5 --scope local --client-id 3

    # N-BaIoT centralized, main 45-client scheme
    python scripts/train_baseline.py --dataset nbaiot --scheme main_45 --scope centralized

    # smoke test: subset the TRAIN split only, few epochs
    python scripts/train_baseline.py --dataset cicids2017 --alpha 5 --scope centralized \
        --epochs 1 --max-train-sequences 2000 --run-tag smoketest

Test evaluation happens exactly once, after training, using the best
(lowest validation loss) checkpoint -- never used for tuning.
Zero-day holdout is reported as reconstruction MSE + label composition
only; Phase 4 does not attempt zero-day classification (Phase 7).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from fedpda_ids.data.sequence_dataset import (  # noqa: E402
    ZeroDaySequenceDataset,
    build_scope_dataloaders,
    check_client_trainable,
)
from fedpda_ids.evaluation.metrics import compute_classification_metrics, extract_rare_class_metrics  # noqa: E402
from fedpda_ids.models.lstm_autoencoder import LSTMAutoencoderClassifier  # noqa: E402
from fedpda_ids.models.trainer import evaluate, load_checkpoint, select_device, train_model  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402


def resolve_scope(config: dict, dataset: str, alpha: str | None, scheme: str | None) -> tuple[Path, str, dict]:
    if dataset == "cicids2017":
        if alpha is None:
            raise ValueError("--alpha is required for --dataset cicids2017")
        seq_dir = Path(config["data"]["cicids2017"]["processed_dir"]) / "sequences" / f"alpha_{alpha}"
        client_id_col = f"client_id_alpha{alpha}"
        rare_labels = config["data"]["cicids2017"]["rare_labels"]
    elif dataset == "nbaiot":
        scheme = scheme or "main_45"
        seq_dir = Path(config["data"]["nbaiot"]["processed_dir"]) / "sequences" / scheme
        client_id_col = "client_id_45" if scheme == "main_45" else "client_id_9"
        rare_labels = config["data"]["nbaiot"]["rare_labels"]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    if not (seq_dir / "X.npy").exists():
        raise FileNotFoundError(f"Sequences not built yet: {seq_dir}")

    return seq_dir, client_id_col, rare_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 centralized/local baseline")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--scope", choices=["centralized", "local"], required=True)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None, help="subset TRAIN only, for smoke tests")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=None,
                         help="override config's project.seed, e.g. for Phase 12's 3-seed final runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    if args.scope == "local" and args.client_id is None:
        parser.error("--client-id is required when --scope local")

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="train_baseline",
    )

    seq_dir, client_id_col, rare_labels = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")
    client_desc = "centralized" if args.scope == "centralized" else f"client{args.client_id}"
    run_name = f"{args.dataset}_{scope_desc}_{client_desc}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"

    results_dir = Path(config["training"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{results_path} already exists -- pass --run-tag to name this run distinctly, "
            "or --overwrite if you intend to replace it."
        )

    client_id = None if args.scope == "centralized" else args.client_id
    if args.scope == "local":
        ok, reason = check_client_trainable(
            seq_dir, client_id_col, client_id,
            config["training"]["local_training"]["min_train_sequences"],
            config["training"]["local_training"]["min_train_classes"],
        )
        logger.info("client %d trainable check: %s (%s)", client_id, ok, reason)
        if not ok:
            print(f"SKIPPED: client {client_id} is not trainable ({reason})")
            summary = {"run_name": run_name, "skipped": True, "reason": reason}
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return

    device = select_device()
    # pin_memory only helps when transferring to a CUDA device -- follow
    # the actually-selected device, not a static config flag, per the
    # frozen "only if appropriate for the actual device configuration" rule.
    pin_memory = device.type == "cuda"

    batch_size = config["model"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, batch_size, num_workers, pin_memory)

    train_dataset = scope["datasets"]["train"]
    if args.max_train_sequences is not None and len(train_dataset) > args.max_train_sequences:
        logger.warning(
            "SMOKE TEST: subsetting TRAIN to %d/%d sequences (val/test untouched) -- "
            "do not treat this run's metrics as the final baseline result",
            args.max_train_sequences, len(train_dataset),
        )
        rng = np.random.default_rng(seed)
        subset_idx = rng.choice(len(train_dataset), size=args.max_train_sequences, replace=False)
        train_dataset = Subset(train_dataset, subset_idx.tolist())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    num_features = next(iter(train_loader))[0].shape[-1]
    num_classes = len(scope["label_to_index"])
    logger.info(
        "run=%s num_features=%d num_classes=%d train=%d val=%d test=%d",
        run_name, num_features, num_classes, len(train_dataset),
        len(scope["datasets"]["val"]), len(scope["datasets"]["test"]),
    )

    model_cfg = config["model"]
    model = LSTMAutoencoderClassifier(
        num_features=num_features,
        num_classes=num_classes,
        window_size=config["sequence"]["window_size"],
        latent_dim=model_cfg["latent_dim"],
        encoder_layer1_units=model_cfg["encoder"]["layer1_units"],
        decoder_layer1_units=model_cfg["decoder"]["layer1_units"],
        classifier_hidden_units=model_cfg["classifier_head"]["hidden_units"],
        classifier_dropout=model_cfg["classifier_head"]["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["optimizer"]["learning_rate"])

    epochs = args.epochs or config["training"]["epochs"]
    run_config = {
        "run_name": run_name,
        "dataset": args.dataset,
        "alpha": args.alpha,
        "scheme": args.dataset == "nbaiot" and (args.scheme or "main_45") or None,
        "scope": args.scope,
        "client_id": client_id,
        "seed": seed,
        "num_features": num_features,
        "num_classes": num_classes,
        "label_to_index": scope["label_to_index"],
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": model_cfg["optimizer"]["learning_rate"],
        "lambda_ce": model_cfg["loss"]["lambda_ce"],
        "smoke_test_max_train_sequences": args.max_train_sequences,
        "unseen_eval_labels": scope["unseen_eval_labels"],
    }

    result = train_model(
        model, train_loader, scope["loaders"]["val"], optimizer, device, epochs,
        model_cfg["loss"]["lambda_ce"], scope["index_to_label"],
        config["training"]["checkpoint_dir"], run_name, run_config,
    )

    # Test evaluation exactly once, using the BEST (lowest val loss) checkpoint.
    best_state = load_checkpoint(result["best_checkpoint"])["model_state_dict"]
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, scope["loaders"]["test"], device, model_cfg["loss"]["lambda_ce"], scope["index_to_label"])
    rare_class_metrics = extract_rare_class_metrics(test_metrics, rare_labels)

    # Zero-day holdout: reconstruction MSE + label composition only (no classification -- Phase 7).
    zero_day_report = None
    zero_day_dataset = ZeroDaySequenceDataset(seq_dir)
    if len(zero_day_dataset) > 0:
        zero_day_loader = DataLoader(
            zero_day_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        model.eval()
        mse_sum, n_batches = 0.0, 0
        label_counts: dict[str, int] = {}
        with torch.no_grad():
            for x, labels in zero_day_loader:
                x = x.to(device)
                reconstruction, _, _ = model(x)
                mse_sum += torch.nn.functional.mse_loss(reconstruction, x).item()
                n_batches += 1
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
        zero_day_report = {
            "num_sequences": len(zero_day_dataset),
            "reconstruction_mse": mse_sum / max(n_batches, 1),
            "label_counts": label_counts,
            "note": "reconstruction MSE only -- classification not attempted (Phase 7 scope)",
        }

    summary = {
        "run_name": run_name,
        "skipped": False,
        "run_config": run_config,
        "history": result["history"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "best_checkpoint": result["best_checkpoint"],
        "last_checkpoint": result["last_checkpoint"],
        "test_metrics": test_metrics,
        "rare_class_test_metrics": rare_class_metrics,
        "zero_day_holdout_report": zero_day_report,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved results to %s", results_path)
    print(f"DONE: {run_name}")
    print(f"  test accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"  results: {results_path}")
    print(f"  best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
