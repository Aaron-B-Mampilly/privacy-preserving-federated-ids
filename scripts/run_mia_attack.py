"""Phase 11: loss-threshold membership inference attack (MIA) against an
already-completed FL run's checkpoint+per-client heads -- validates
empirically whether Phase 8's DP mechanism (or Phase 9's SecAgg+, or
Phase 6's plain baseline) actually limits what an attacker can infer
about training-set membership, not just in theory.

Per-client: member = that client's own TRAIN split (used to train both
the shared encoder/decoder and its own head), non-member = that SAME
client's own TEST split (same client, structurally held out, never
trained on).

Usage:
    # attack a non-DP personalized run
    python scripts/run_mia_attack.py --dataset cicids2017 --alpha 5 \
        --source-run-name cicids2017_5_personalized --checkpoint-suffix best

    # attack one point of the DP epsilon sweep
    python scripts/run_mia_attack.py --dataset cicids2017 --alpha 5 \
        --source-run-name cicids2017_5_dp_personalized_eps8.0 --checkpoint-suffix last
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from fedpda_ids.data.sequence_dataset import build_scope_dataloaders  # noqa: E402
from fedpda_ids.federated.client import load_local_head, local_head_path  # noqa: E402
from fedpda_ids.federated.simulation import load_shared_checkpoint, make_model  # noqa: E402
from fedpda_ids.models.trainer import select_device  # noqa: E402
from fedpda_ids.privacy.attacks import compute_per_example_losses, run_loss_threshold_mia  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11: loss-threshold MIA")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--source-run-name", type=str, required=True,
                         help="exact run_name whose checkpoint+per-client heads to attack, "
                              "e.g. cicids2017_5_personalized")
    parser.add_argument("--checkpoint-suffix", choices=["best", "last"], default="best")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="run_mia_attack",
    )

    seq_dir, client_id_col, _ = resolve_scope(config, args.dataset, args.alpha, args.scheme)

    results_dir = Path(config["training"]["results_dir"])
    source_results_path = results_dir / f"{args.source_run_name}.json"
    if not source_results_path.exists():
        raise FileNotFoundError(
            f"{source_results_path} not found -- run the corresponding training script first "
            f"(expected run_name={args.source_run_name})."
        )
    source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    label_to_index = source_results["run_config"]["label_to_index"]
    # The saved training-script summaries don't keep a top-level "pool" list --
    # per_client_test_metrics's keys are exactly the pool clients that were
    # actually evaluated (had a saved head); a client with no saved head can't
    # be attacked anyway (no personalized model exists for it), so this is
    # the correct set, not an approximation.
    pool = [int(k) for k in source_results["per_client_test_metrics"].keys()]

    run_name = f"mia_{args.source_run_name}_{args.checkpoint_suffix}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists -- pass --run-tag or --overwrite.")

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    shared_ckpt_path = checkpoint_dir / f"{args.source_run_name}_{args.checkpoint_suffix}.pt"
    if not shared_ckpt_path.exists():
        raise FileNotFoundError(f"{shared_ckpt_path} not found.")

    device = select_device()
    batch_size = config["model"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    model_cfg = config["model"]
    window_size = config["sequence"]["window_size"]
    lambda_ce = model_cfg["loss"]["lambda_ce"]
    num_classes = len(label_to_index)

    centralized_scope = build_scope_dataloaders(seq_dir, client_id_col, None, batch_size, num_workers, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]

    shared_model = make_model(num_features, num_classes, model_cfg, window_size)
    load_shared_checkpoint(shared_ckpt_path, shared_model)

    per_client_results = {}
    skipped = []
    all_member, all_non_member = [], []
    for client_id in pool:
        head_path = local_head_path(checkpoint_dir, args.source_run_name, client_id)
        client_model = make_model(num_features, num_classes, model_cfg, window_size).to(device)
        client_model.load_state_dict(shared_model.state_dict())
        if not load_local_head(client_model, head_path):
            skipped.append(client_id)
            continue

        scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, batch_size, num_workers, False, label_to_index)
        if len(scope["datasets"]["test"]) == 0:
            per_client_results[client_id] = {"status": "empty_test_split"}
            continue

        member_losses = compute_per_example_losses(client_model, scope["loaders"]["train"], device, lambda_ce)
        non_member_losses = compute_per_example_losses(client_model, scope["loaders"]["test"], device, lambda_ce)
        per_client_results[client_id] = run_loss_threshold_mia(member_losses, non_member_losses)
        all_member.append(member_losses)
        all_non_member.append(non_member_losses)

    if skipped:
        logger.warning("%d/%d pool clients skipped (no saved head): %s", len(skipped), len(pool), skipped)

    pooled_member = np.concatenate(all_member) if all_member else np.array([])
    pooled_non_member = np.concatenate(all_non_member) if all_non_member else np.array([])
    pooled_result = run_loss_threshold_mia(pooled_member, pooled_non_member)

    valid = [r for r in per_client_results.values() if "status" not in r and not np.isnan(r["auc"])]
    per_client_summary = {
        "num_clients_evaluated": len(valid),
        "num_clients_skipped_no_head": len(skipped),
        "auc_mean": float(np.mean([r["auc"] for r in valid])) if valid else None,
        "auc_std": float(np.std([r["auc"] for r in valid])) if valid else None,
        "advantage_mean": float(np.mean([r["advantage"] for r in valid])) if valid else None,
        "advantage_std": float(np.std([r["advantage"] for r in valid])) if valid else None,
    }

    summary = {
        "run_name": run_name,
        "source_run_name": args.source_run_name,
        "checkpoint_suffix": args.checkpoint_suffix,
        "run_config": {"dataset": args.dataset, "alpha": args.alpha, "scheme": args.scheme, "pool_size": len(pool)},
        "pooled_result": pooled_result,
        "per_client_summary": per_client_summary,
        "per_client_results": {str(k): v for k, v in per_client_results.items()},
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved results to %s", results_path)
    print(f"DONE: {run_name}")
    print(f"  pooled AUC={pooled_result['auc']:.4f} advantage={pooled_result['advantage']:.4f} "
          f"(member_loss={pooled_result['mean_member_loss']:.4f} non_member_loss={pooled_result['mean_non_member_loss']:.4f})")
    if per_client_summary["auc_mean"] is not None:
        print(f"  per-client: n={per_client_summary['num_clients_evaluated']} "
              f"AUC={per_client_summary['auc_mean']:.4f}±{per_client_summary['auc_std']:.4f} "
              f"advantage={per_client_summary['advantage_mean']:.4f}±{per_client_summary['advantage_std']:.4f}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
