"""Phase 6: Personalized FL (shared encoder/decoder, local classifier head).

Usage:
    # smoke test: few rounds, before committing to a full 100-round run
    python scripts/train_personalized.py --dataset cicids2017 --alpha 5 --rounds 3 --run-tag smoketest

    # full run
    python scripts/train_personalized.py --dataset cicids2017 --alpha 5
    python scripts/train_personalized.py --dataset nbaiot --scheme main_45
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpda_ids.federated.simulation import run_personalized_simulation  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Personalized FL")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="train_personalized",
    )

    seq_dir, client_id_col, _ = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")
    run_name = f"{args.dataset}_{scope_desc}_personalized"
    if args.run_tag:
        run_name += f"_{args.run_tag}"

    results_dir = Path(config["training"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists -- pass --run-tag or --overwrite.")

    fed_cfg = config["federated"]
    if args.dataset == "cicids2017":
        num_clients_configured = fed_cfg["cicids2017"]["num_clients"]
        clients_per_round = fed_cfg["cicids2017"]["clients_per_round"]
    else:
        scheme = args.scheme or "main_45"
        num_clients_configured = fed_cfg["nbaiot"]["num_clients"] if scheme == "main_45" else fed_cfg["nbaiot"]["ablation_num_clients"]
        clients_per_round = max(1, round(num_clients_configured * fed_cfg["client_participation_fraction"]))

    num_rounds = args.rounds or fed_cfg["num_rounds"]

    logger.info(
        "Starting personalized FL: run=%s clients=%d/round clients_configured=%d rounds=%d",
        run_name, clients_per_round, num_clients_configured, num_rounds,
    )

    t0 = time.time()
    result = run_personalized_simulation(
        seq_dir=seq_dir, client_id_col=client_id_col, num_clients_configured=num_clients_configured,
        clients_per_round=clients_per_round, num_rounds=num_rounds,
        local_epochs=fed_cfg["local_epochs"], batch_size=config["model"]["batch_size"],
        model_cfg=config["model"], window_size=config["sequence"]["window_size"],
        min_train_sequences=config["training"]["local_training"]["min_train_sequences"],
        min_train_classes=config["training"]["local_training"]["min_train_classes"],
        checkpoint_dir=config["training"]["checkpoint_dir"], run_name=run_name, seed=seed,
        num_workers=config["training"]["num_workers"],
    )
    elapsed = time.time() - t0

    summary = {
        "run_name": run_name, "elapsed_seconds": elapsed,
        "run_config": result["run_config"],
        "history_losses_centralized": result["history_losses_centralized"],
        "history_metrics_centralized": result["history_metrics_centralized"],
        "history_losses_distributed": result["history_losses_distributed"],
        "history_metrics_distributed_fit": result["history_metrics_distributed_fit"],
        "history_metrics_distributed": result["history_metrics_distributed"],
        "best_round": result["best_round"], "best_val_mse": result["best_val_mse"],
        "per_client_test_metrics": {str(k): v for k, v in result["per_client_test_metrics"].items()},
        "per_client_summary": result["per_client_summary"],
        "best_checkpoint": result["best_checkpoint"], "last_checkpoint": result["last_checkpoint"],
        "trainable_pool_size": len(result["pool"]), "num_clients_configured": num_clients_configured,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Done in %.1fs. Saved results to %s", elapsed, results_path)
    summ = result["per_client_summary"]
    print(f"DONE: {run_name} ({elapsed:.1f}s)")
    print(f"  best round={result['best_round']} best_val_mse={result['best_val_mse']:.4f}")
    print(f"  per-client: n={summ['num_clients_evaluated']} skipped={summ['num_clients_skipped_no_head']}")
    print(f"  accuracy: mean={summ['accuracy_mean']:.4f} std={summ['accuracy_std']:.4f}")
    print(f"  macro_f1: mean={summ['macro_f1_mean']:.4f} std={summ['macro_f1_std']:.4f}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
