"""Phase 7: latent class prototypes for zero-day/rare-class detection (E4).

Post-training pipeline built on top of an ALREADY-COMPLETED Phase 6
personalized-FL run: loads that run's best shared-encoder checkpoint,
has every pool client compute its own protected (clipped) per-class
prototypes from its own TRAIN split, aggregates them server-side, and
calibrates the NEW-CLASS distance threshold from the aggregated
prototypes themselves (tau = support-weighted mean pairwise distance *
multiplier -- the frozen formula, see configs/config.yaml). Evaluation
uses the real CICIDS2017/N-BaIoT zero-day holdout (true novel classes,
expect -1) and the centralized TEST split (true known classes, exactly
once, never used to tune anything here -- false positive rate only).

DP noise on the transmitted prototypes is Phase 8 scope -- this script
calls client_prototypes(..., noise_fn=None), i.e. epsilon=infinity.

Usage:
    # smoke test against a completed personalized run
    python scripts/compute_prototypes.py --dataset cicids2017 --alpha 5 --run-tag smoketest

    # full run (requires experiments/checkpoints/cicids2017_5_personalized_best.pt to exist)
    python scripts/compute_prototypes.py --dataset cicids2017 --alpha 5
    python scripts/compute_prototypes.py --dataset nbaiot --scheme main_45
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fedpda_ids.data.sequence_dataset import (  # noqa: E402
    ZeroDaySequenceDataset,
    build_label_index,
    build_scope_dataloaders,
    get_scope_train_labels,
)
from fedpda_ids.evaluation.metrics import compute_zero_day_metrics  # noqa: E402
from fedpda_ids.federated.simulation import build_trainable_client_pool, load_shared_checkpoint, make_model  # noqa: E402
from fedpda_ids.privacy.dp import calibrate_prototype_noise_std  # noqa: E402
from fedpda_ids.models.prototypes import (  # noqa: E402
    aggregate_prototypes,
    calibrate_threshold,
    classify_batch_with_prototypes,
    client_prototypes,
    extract_latents,
)
from fedpda_ids.models.trainer import select_device  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


@torch.no_grad()
def _extract_latents_with_string_labels(encoder, loader: DataLoader, device: torch.device):
    """Like prototypes.extract_latents, but for ZeroDaySequenceDataset's
    (x, label_string) items -- zero-day labels are never in any scope's
    label_to_index, so they can't be represented as the int tensor
    extract_latents() expects."""
    encoder.eval()
    all_z, all_labels = [], []
    for x, labels in loader:
        x = x.to(device)
        z = encoder(x)
        all_z.append(z.cpu().numpy())
        all_labels.extend(labels)
    if not all_z:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)
    return np.concatenate(all_z), np.array(all_labels, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: latent class prototypes")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--personalized-run-tag", type=str, default="",
                         help="run-tag the Phase 6 train_personalized.py run was saved under, if any")
    parser.add_argument("--clip-bound", type=float, default=None)
    parser.add_argument("--threshold-multiplier", type=float, default=None)
    parser.add_argument("--target-epsilon", type=float, default=None,
                         help="Phase 8: DP noise for the transmitted prototypes. Default (unset) = "
                              "inf = no noise, the original Phase 7 behavior.")
    parser.add_argument("--target-delta", type=float, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="compute_prototypes",
    )

    seq_dir, client_id_col, rare_labels = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")

    personalized_run_name = f"{args.dataset}_{scope_desc}_personalized"
    if args.personalized_run_tag:
        personalized_run_name += f"_{args.personalized_run_tag}"

    run_name = f"{args.dataset}_{scope_desc}_prototypes"
    if args.run_tag:
        run_name += f"_{args.run_tag}"

    results_dir = Path(config["training"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists -- pass --run-tag or --overwrite.")

    best_shared_ckpt = Path(config["training"]["checkpoint_dir"]) / f"{personalized_run_name}_best.pt"
    if not best_shared_ckpt.exists():
        raise FileNotFoundError(
            f"{best_shared_ckpt} not found -- run scripts/train_personalized.py for this scope first "
            f"(expected run_name={personalized_run_name})."
        )

    proto_cfg = config["prototypes"]
    clip_bound = args.clip_bound if args.clip_bound is not None else proto_cfg["clip_bound"]
    threshold_multiplier = args.threshold_multiplier if args.threshold_multiplier is not None else proto_cfg["zero_day_threshold_multiplier"]
    target_epsilon = args.target_epsilon if args.target_epsilon is not None else float("inf")
    target_delta = args.target_delta if args.target_delta is not None else config["privacy"]["delta"]

    # Same global class vocabulary + trainable pool Phase 6 used, recomputed
    # deterministically from the same sequence artifacts/config (not parsed
    # from Phase 6's results json, so this has no file-format coupling).
    fed_cfg = config["federated"]
    if args.dataset == "cicids2017":
        num_clients_configured = fed_cfg["cicids2017"]["num_clients"]
    else:
        scheme = args.scheme or "main_45"
        num_clients_configured = fed_cfg["nbaiot"]["num_clients"] if scheme == "main_45" else fed_cfg["nbaiot"]["ablation_num_clients"]

    global_labels = get_scope_train_labels(seq_dir, client_id_col, None)
    label_to_index = build_label_index(global_labels)
    index_to_label = {i: label for label, i in label_to_index.items()}
    num_classes = len(label_to_index)

    pool = build_trainable_client_pool(
        seq_dir, client_id_col, num_clients_configured,
        config["training"]["local_training"]["min_train_sequences"],
        config["training"]["local_training"]["min_train_classes"],
    )
    logger.info("Prototype pipeline: %d/%d configured clients are trainable", len(pool), num_clients_configured)

    device = select_device()
    batch_size = config["model"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    model_cfg = config["model"]
    window_size = config["sequence"]["window_size"]

    centralized_scope = build_scope_dataloaders(seq_dir, client_id_col, None, batch_size, num_workers, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]

    model = make_model(num_features, num_classes, model_cfg, window_size).to(device)
    load_shared_checkpoint(best_shared_ckpt, model)
    encoder = model.encoder

    # Phase 8: DP noise for the transmitted prototypes. target_epsilon=inf
    # (default) means noise_std=0.0 -- unchanged Phase 7 behavior. See
    # calibrate_prototype_noise_std()'s docstring for why the joint
    # sensitivity across this client's (up to num_classes) contributed
    # prototypes -- not clip_bound alone -- is what's calibrated against.
    noise_std = calibrate_prototype_noise_std(target_epsilon, target_delta, clip_bound, num_classes)
    noise_rng = np.random.default_rng(seed)
    noise_fn = (lambda v: v + noise_rng.normal(0.0, noise_std, size=v.shape)) if noise_std > 0.0 else None

    # 1. Each pool client computes its own protected prototypes from its
    #    OWN train split (never centralized/pooled raw data).
    client_prototype_list = []
    per_client_report = {}
    for client_id in pool:
        scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, batch_size, num_workers, False, label_to_index)
        protos = client_prototypes(encoder, scope["loaders"]["train"], device, clip_bound=clip_bound, noise_fn=noise_fn)
        client_prototype_list.append(protos)
        per_client_report[client_id] = {
            "num_classes": len(protos),
            "support_by_class": {index_to_label[label]: count for label, (_, count) in protos.items()},
        }

    # 2. Server-side aggregation (support-weighted mean).
    aggregated, support = aggregate_prototypes(client_prototype_list, return_support=True)

    missing_classes = set(label_to_index.values()) - set(aggregated.keys())
    if missing_classes:
        logger.warning(
            "%d/%d global-vocab classes have NO prototype (every contributing client was excluded "
            "from the trainable pool): %s",
            len(missing_classes), num_classes, sorted(index_to_label[c] for c in missing_classes),
        )

    # 3. Threshold calibration: formula-only, from the aggregated
    #    train-derived prototypes + their total support -- never from
    #    test/zero-day data (see prototypes.calibrate_threshold docstring).
    threshold = calibrate_threshold(aggregated, support, multiplier=threshold_multiplier)

    # 4. Evaluation.
    #    a) TRUE zero-day holdout -- expect -1 (NEW CLASS).
    zero_day_dataset = ZeroDaySequenceDataset(seq_dir)
    zero_day_loader = DataLoader(zero_day_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    z_zero_day, labels_zero_day = _extract_latents_with_string_labels(encoder, zero_day_loader, device)
    if len(z_zero_day):
        # clip_bound=clip_bound: PRE-CODING CORRECTION (approved) -- query
        # latents are clipped the same way prototypes were, so distance
        # comparisons aren't confounded by the encoder's raw latent norm
        # (measured ~2.28 vs B=1.0 on real CICIDS2017 data). See
        # prototypes.classify_batch_with_prototypes docstring.
        zero_day_preds, zero_day_dists = classify_batch_with_prototypes(z_zero_day, aggregated, threshold, clip_bound=clip_bound)
    else:
        zero_day_preds, zero_day_dists = np.array([]), np.array([])

    per_zero_day_label = {}
    for label in np.unique(labels_zero_day) if len(labels_zero_day) else []:
        mask = labels_zero_day == label
        per_zero_day_label[str(label)] = {
            "num_sequences": int(mask.sum()),
            "detection_rate": float((zero_day_preds[mask] == -1).mean()),
        }

    #    b) TRUE known classes (centralized TEST split, evaluated exactly
    #       once) -- expect NOT -1; false positives measured here.
    z_known, y_known = extract_latents(encoder, centralized_scope["loaders"]["test"], device)
    if len(z_known):
        known_preds, known_dists = classify_batch_with_prototypes(z_known, aggregated, threshold, clip_bound=clip_bound)
    else:
        known_preds, known_dists = np.array([]), np.array([])

    zero_day_metrics = compute_zero_day_metrics(zero_day_preds, known_preds)

    summary = {
        "run_name": run_name,
        "personalized_run_name": personalized_run_name,
        "run_config": {
            "dataset": args.dataset, "alpha": args.alpha, "scheme": scope_desc,
            "clip_bound": clip_bound, "threshold_multiplier": threshold_multiplier,
            "target_epsilon": target_epsilon, "target_delta": target_delta, "prototype_noise_std": noise_std,
            "seed": seed, "num_features": num_features, "num_classes": num_classes,
            "label_to_index": label_to_index, "pool_size": len(pool), "num_clients_configured": num_clients_configured,
        },
        "threshold": threshold,
        "global_prototype_support": {index_to_label[label]: count for label, count in support.items()},
        "missing_classes": sorted(index_to_label[c] for c in missing_classes),
        "per_client_report": {str(k): v for k, v in per_client_report.items()},
        "zero_day_metrics": zero_day_metrics,
        "per_zero_day_label_detection": per_zero_day_label,
        "num_known_test_sequences": int(len(z_known)),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved results to %s", results_path)
    print(f"DONE: {run_name}")
    print(f"  pool={len(pool)}/{num_clients_configured} classes_with_prototypes={len(aggregated)}/{num_classes} threshold={threshold:.4f}")
    print(f"  zero-day detection_rate={zero_day_metrics['zero_day_detection_rate']:.4f} "
          f"false_positive_rate={zero_day_metrics['false_positive_rate']:.4f} f1={zero_day_metrics['f1']:.4f}")
    for label, stats in per_zero_day_label.items():
        print(f"    {label}: n={stats['num_sequences']} detection_rate={stats['detection_rate']:.4f}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
