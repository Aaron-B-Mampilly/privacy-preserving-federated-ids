"""E5's live drift-triggered retraining -- Phase 10 built DETECTION only
(explicit kickoff scope decision: "these functions DETECT and REPORT
where the trigger condition would fire; actually re-training on trigger
is deferred"). This script closes that gap and produces E5's Table T5's
four real numbers for one scope: F1 before drift, F1 after drift
without adaptation, drift detection delay in rounds, F1 after triggered
retraining.

Reuses, never recomputes: Phase 10's own completed `{scope}_drift.json`
(round_window_size, first_trigger_round) and Phase 6's own completed
personalized-FL checkpoint + per-client heads -- the ONLY new compute
here is (a) recovering the real cutoff timestamp Phase 10's trigger
round corresponds to, (b) evaluating the existing checkpoint on the
val split (a forward pass Phase 6 never separately reported) and on
the post-cutoff held-out slice of each client's own test split, and
(c) actually executing the retrain (run_drift_triggered_retraining).

"Without the drift monitor" (T5's spec) = the "no adaptation" arm
(without a monitor, nothing would ever have triggered a retrain, so
the deployed model just keeps serving the original frozen checkpoint);
"with the drift monitor" = the "after triggered retraining" arm.

Usage:
    python scripts/run_drift_triggered_retraining.py --dataset cicids2017 --alpha 5 \
        --num-retrain-rounds 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fedpda_ids.data.sequence_dataset import LabeledSequenceIndexDataset, build_scope_dataloaders  # noqa: E402
from fedpda_ids.drift.retrain import split_client_rows_at_cutoff, trigger_cutoff_timestamp  # noqa: E402
from fedpda_ids.federated.client import load_local_head, local_head_path  # noqa: E402
from fedpda_ids.federated.simulation import (  # noqa: E402
    build_trainable_client_pool,
    load_shared_checkpoint,
    make_model,
    run_drift_triggered_retraining,
)
from fedpda_ids.models.trainer import evaluate, select_device  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


def _evaluate_pool(pool, checkpoint_dir, run_name, seq_dir_scope_by_client, model_builder, lambda_ce, index_to_label, device):
    """Evaluates `run_name`'s current shared checkpoint + each client's
    OWN persisted head against `seq_dir_scope_by_client[client_id]`
    (an arbitrary caller-supplied DataLoader per client -- val loader,
    eval_loader, whatever). Skips (never fabricates) a client with no
    saved head or an empty loader."""
    shared_model = model_builder()
    load_shared_checkpoint(Path(checkpoint_dir) / f"{run_name}_best.pt", shared_model)

    per_client = {}
    for client_id in pool:
        loader = seq_dir_scope_by_client.get(client_id)
        if loader is None or len(loader.dataset) == 0:
            per_client[client_id] = {"status": "no_eval_data"}
            continue
        client_model = model_builder()
        client_model.load_state_dict(shared_model.state_dict())
        if not load_local_head(client_model, local_head_path(checkpoint_dir, run_name, client_id)):
            per_client[client_id] = {"status": "no_saved_head"}
            continue
        per_client[client_id] = evaluate(client_model, loader, device, lambda_ce, index_to_label)
    return per_client


def _summarize(per_client: dict) -> dict:
    valid = [m for m in per_client.values() if "status" not in m]
    macro_f1s = [m["macro_f1"] for m in valid]
    return {
        "num_clients_evaluated": len(valid),
        "macro_f1_mean": float(np.mean(macro_f1s)) if macro_f1s else None,
        "macro_f1_std": float(np.std(macro_f1s)) if macro_f1s else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: live drift-triggered retraining")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--personalized-run-name", type=str, default=None,
                         help="defaults to {dataset}_{scope}_personalized")
    parser.add_argument("--drift-run-name", type=str, default=None,
                         help="defaults to {dataset}_{scope}_drift (Phase 10's saved result)")
    parser.add_argument("--num-retrain-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=None, help="defaults to config's federated.local_epochs")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="run_drift_triggered_retraining",
    )

    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")
    personalized_run_name = args.personalized_run_name or f"{args.dataset}_{scope_desc}_personalized"
    drift_run_name = args.drift_run_name or f"{args.dataset}_{scope_desc}_drift"

    run_name = f"{args.dataset}_{scope_desc}_drift_retrain"
    if args.run_tag:
        run_name += f"_{args.run_tag}"

    results_dir = Path(config["training"]["results_dir"])
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists -- pass --run-tag or --overwrite.")

    personalized_results = json.loads((results_dir / f"{personalized_run_name}.json").read_text(encoding="utf-8"))
    drift_results = json.loads((results_dir / f"{drift_run_name}.json").read_text(encoding="utf-8"))

    label_to_index = personalized_results["run_config"]["label_to_index"]
    index_to_label = {i: label for label, i in label_to_index.items()}
    num_classes = len(label_to_index)
    round_window_size = drift_results["run_config"]["round_window_size"]
    first_trigger_round = drift_results["first_trigger_round"]
    if first_trigger_round is None:
        raise ValueError(f"{drift_run_name} never triggered a retrain -- nothing for this script to do.")

    seq_dir, client_id_col, _ = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    fed_cfg = config["federated"]
    local_epochs = args.local_epochs or fed_cfg["local_epochs"]
    model_cfg = config["model"]
    window_size = config["sequence"]["window_size"]
    lambda_ce = model_cfg["loss"]["lambda_ce"]
    learning_rate = model_cfg["optimizer"]["learning_rate"]
    batch_size = model_cfg["batch_size"]
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    device = select_device()

    min_seq = config["training"]["local_training"]["min_train_sequences"]
    min_cls = config["training"]["local_training"]["min_train_classes"]
    num_clients_configured = personalized_results["run_config"]["num_clients_configured"]
    pool = build_trainable_client_pool(seq_dir, client_id_col, num_clients_configured, min_seq, min_cls)

    centralized_scope = build_scope_dataloaders(seq_dir, client_id_col, None, batch_size, 0, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]

    def model_builder():
        return make_model(num_features, num_classes, model_cfg, window_size).to(device)

    # -----------------------------------------------------------------
    # Recover the REAL cutoff timestamp Phase 10's first_trigger_round
    # corresponds to (same pooled, globally-sorted test stream detect_drift.py used).
    # -----------------------------------------------------------------
    cols_needed = ["temporal_split", "sequence_label", "sequence_index", "window_start_time", client_id_col]
    metadata = pd.read_parquet(seq_dir / "metadata.parquet", columns=cols_needed)
    pooled_test = metadata[(metadata["temporal_split"] == "test") & (metadata[client_id_col] != -1)].sort_values("window_start_time")
    cutoff_timestamp = trigger_cutoff_timestamp(
        pooled_test["window_start_time"].to_numpy(), round_window_size=round_window_size, first_trigger_round=first_trigger_round,
    )
    logger.info("Real drift-trigger cutoff timestamp: %s (round %d, window_size %d)", cutoff_timestamp, first_trigger_round, round_window_size)

    # -----------------------------------------------------------------
    # Per-client val loader ("before drift") + retrain/eval split of this
    # client's own test-split rows at the real cutoff.
    # -----------------------------------------------------------------
    val_loaders, retrain_loaders, eval_loaders = {}, {}, {}
    for client_id in pool:
        scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, batch_size, 0, False, label_to_index)
        val_loaders[client_id] = scope["loaders"]["val"]

        client_test_rows = metadata[(metadata["temporal_split"] == "test") & (metadata[client_id_col] == client_id)]
        retrain_rows, eval_rows = split_client_rows_at_cutoff(client_test_rows, "window_start_time", cutoff_timestamp)

        retrain_ds = LabeledSequenceIndexDataset(seq_dir, retrain_rows["sequence_index"].to_numpy(), retrain_rows["sequence_label"].to_numpy(), label_to_index)
        eval_ds = LabeledSequenceIndexDataset(seq_dir, eval_rows["sequence_index"].to_numpy(), eval_rows["sequence_label"].to_numpy(), label_to_index)
        # shuffle=True on an EMPTY dataset raises at construction time
        # (RandomSampler requires num_samples > 0) -- a real client can
        # genuinely have zero rows on one side of the cutoff (found via a
        # real N-BaIoT run), and that client is meant to be skipped
        # downstream (run_drift_triggered_retraining's own len()==0 check),
        # never crash the whole script before it gets the chance to.
        retrain_loaders[client_id] = DataLoader(retrain_ds, batch_size=batch_size, shuffle=len(retrain_ds) > 0)
        eval_loaders[client_id] = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    # -----------------------------------------------------------------
    # 1) F1 before drift, 2) F1 after drift without adaptation (= "without drift monitor").
    # -----------------------------------------------------------------
    before_per_client = _evaluate_pool(pool, checkpoint_dir, personalized_run_name, val_loaders, model_builder, lambda_ce, index_to_label, device)
    no_adapt_per_client = _evaluate_pool(pool, checkpoint_dir, personalized_run_name, eval_loaders, model_builder, lambda_ce, index_to_label, device)

    # -----------------------------------------------------------------
    # 3) Actually retrain on trigger (= "with drift monitor"), then re-evaluate on the SAME held-out eval slice.
    # -----------------------------------------------------------------
    retrain_run_name = f"{personalized_run_name}_drift_retrained"
    if args.run_tag:
        retrain_run_name += f"_{args.run_tag}"
    retrain_result = run_drift_triggered_retraining(
        pool=pool, retrain_loaders=retrain_loaders, checkpoint_dir=checkpoint_dir,
        source_run_name=personalized_run_name, new_run_name=retrain_run_name,
        num_features=num_features, num_classes=num_classes, model_cfg=model_cfg, window_size=window_size,
        local_epochs=local_epochs, num_retrain_rounds=args.num_retrain_rounds,
        lambda_ce=lambda_ce, learning_rate=learning_rate, device=device,
    )
    after_retrain_per_client = _evaluate_pool(pool, checkpoint_dir, retrain_run_name, eval_loaders, model_builder, lambda_ce, index_to_label, device)

    before_summary = _summarize(before_per_client)
    no_adapt_summary = _summarize(no_adapt_per_client)
    after_retrain_summary = _summarize(after_retrain_per_client)

    summary = {
        "run_name": run_name,
        "personalized_run_name": personalized_run_name, "drift_run_name": drift_run_name,
        "run_config": {
            "dataset": args.dataset, "alpha": args.alpha, "scheme": args.scheme,
            "round_window_size": round_window_size, "first_trigger_round": first_trigger_round,
            "cutoff_timestamp": cutoff_timestamp, "num_retrain_rounds_requested": args.num_retrain_rounds,
            "num_retrain_rounds_run": retrain_result["num_retrain_rounds_run"],
            "clients_retrained": retrain_result["clients_retrained"], "pool_size": len(pool),
        },
        "f1_before_drift": before_summary,
        "f1_after_drift_no_adaptation_without_monitor": no_adapt_summary,
        "drift_detection_delay_rounds": first_trigger_round,
        "f1_after_triggered_retraining_with_monitor": after_retrain_summary,
        "per_client": {
            "before_drift": {str(k): v for k, v in before_per_client.items()},
            "after_drift_no_adaptation": {str(k): v for k, v in no_adapt_per_client.items()},
            "after_triggered_retraining": {str(k): v for k, v in after_retrain_per_client.items()},
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved results to %s", results_path)
    print(f"DONE: {run_name}")
    print(f"  F1 before drift:                        {before_summary['macro_f1_mean']}")
    print(f"  F1 after drift, no adaptation (no monitor): {no_adapt_summary['macro_f1_mean']}")
    print(f"  drift detection delay (rounds):         {first_trigger_round}")
    print(f"  F1 after triggered retraining (monitor):   {after_retrain_summary['macro_f1_mean']}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
