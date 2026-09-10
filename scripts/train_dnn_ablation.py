"""Phase 4 Part J: LSTM vs DNN ablation.

Trains the non-temporal DNNBaseline on the SAME scope (dataset, alpha/
scheme, centralized/local, client) and class-index mapping the LSTM
baseline used, so the comparison is apples-to-apples on the same
train/val/test split. Does not touch the LSTM results already saved.

Usage:
    python scripts/train_dnn_ablation.py --dataset cicids2017 --alpha 5 --scope centralized
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from fedpda_ids.data.sequence_dataset import build_scope_dataloaders, check_client_trainable  # noqa: E402
from fedpda_ids.models.dnn_baseline import DNNBaseline  # noqa: E402
from fedpda_ids.models.trainer import evaluate_classifier_only, load_checkpoint, select_device, train_model_classifier_only  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Part J: LSTM vs DNN ablation")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--scope", choices=["centralized", "local"], required=True)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    if args.scope == "local" and args.client_id is None:
        parser.error("--client-id is required when --scope local")

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="train_dnn_ablation",
    )

    seq_dir, client_id_col, rare_labels = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")
    client_desc = "centralized" if args.scope == "centralized" else f"client{args.client_id}"
    run_name = f"{args.dataset}_{scope_desc}_{client_desc}_dnn"

    results_dir = Path(config["training"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists -- pass --overwrite to replace it.")

    client_id = None if args.scope == "centralized" else args.client_id
    if args.scope == "local":
        ok, reason = check_client_trainable(
            seq_dir, client_id_col, client_id,
            config["training"]["local_training"]["min_train_sequences"],
            config["training"]["local_training"]["min_train_classes"],
        )
        if not ok:
            print(f"SKIPPED: client {client_id} is not trainable ({reason})")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"run_name": run_name, "skipped": True, "reason": reason}, f, indent=2)
            return

    device = select_device()
    pin_memory = device.type == "cuda"
    batch_size = config["model"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, batch_size, num_workers, pin_memory)

    num_features = next(iter(scope["loaders"]["train"]))[0].shape[-1]
    num_classes = len(scope["label_to_index"])
    logger.info("DNN ablation run=%s num_features=%d num_classes=%d", run_name, num_features, num_classes)

    model_cfg = config["model"]
    model = DNNBaseline(
        num_features=num_features, num_classes=num_classes,
        hidden1_units=model_cfg["encoder"]["layer1_units"], latent_dim=model_cfg["latent_dim"],
        classifier_hidden_units=model_cfg["classifier_head"]["hidden_units"],
        classifier_dropout=model_cfg["classifier_head"]["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["optimizer"]["learning_rate"])

    epochs = args.epochs or config["training"]["epochs"]
    run_config = {
        "run_name": run_name, "dataset": args.dataset, "alpha": args.alpha, "scope": args.scope,
        "client_id": client_id, "seed": seed, "num_features": num_features, "num_classes": num_classes,
        "label_to_index": scope["label_to_index"], "batch_size": batch_size, "epochs": epochs,
        "model_type": "dnn_nontemporal_baseline",
    }

    result = train_model_classifier_only(
        model, scope["loaders"]["train"], scope["loaders"]["val"], optimizer, device, epochs,
        scope["index_to_label"], config["training"]["checkpoint_dir"], run_name, run_config,
    )

    best_state = load_checkpoint(result["best_checkpoint"])["model_state_dict"]
    model.load_state_dict(best_state)
    test_metrics = evaluate_classifier_only(model, scope["loaders"]["test"], device, scope["index_to_label"])

    summary = {
        "run_name": run_name, "skipped": False, "run_config": run_config,
        "history": result["history"], "best_epoch": result["best_epoch"], "best_val_loss": result["best_val_loss"],
        "best_checkpoint": result["best_checkpoint"], "test_metrics": test_metrics,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"DONE: {run_name}")
    print(f"  test accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
