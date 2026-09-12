"""Phase 10: unsupervised concept-drift detection (ADWIN primary, PSI
backup) on a scope's real chronological test-time stream, using an
already-trained shared encoder/decoder (reuses Phase 6's personalized
FL checkpoint -- this is a post-training monitoring pipeline, not part
of the FL training loop itself).

Reference distribution (frozen spec: "last 5000 benign windows"):
the chronologically LAST `psi_reference_window_size` BENIGN-labeled
sequences from the scope's own TRAIN split -- i.e. "normal" traffic
right up to the point training data ends, representing the baseline
an operator would have calibrated against at deployment time.

Monitored stream: every sequence in the scope's TEST split (any
label), re-ordered by its true chronological window_start_time --
detection is unsupervised, so labels are never used for anything
except reporting composition after the fact.

This script DETECTS and REPORTS where the retrain trigger would fire;
it does not execute a retrain (user decision, Phase 10 kickoff --
deferred to Phase 12's full experiment suite).

Usage:
    python scripts/detect_drift.py --dataset cicids2017 --alpha 5
    python scripts/detect_drift.py --dataset nbaiot --scheme main_45
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fedpda_ids.data.sequence_dataset import SequenceIndexDataset  # noqa: E402
from fedpda_ids.drift.detectors import compute_per_sequence_errors_and_latents, evaluate_drift_stream  # noqa: E402
from fedpda_ids.federated.simulation import load_shared_checkpoint, make_model  # noqa: E402
from fedpda_ids.models.trainer import select_device  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10: unsupervised drift detection")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--personalized-run-tag", type=str, default="",
                         help="run-tag the Phase 6 train_personalized.py run was saved under, if any")
    parser.add_argument("--round-window-size", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="detect_drift",
    )

    seq_dir, client_id_col, _ = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")

    personalized_run_name = f"{args.dataset}_{scope_desc}_personalized"
    if args.personalized_run_tag:
        personalized_run_name += f"_{args.personalized_run_tag}"

    run_name = f"{args.dataset}_{scope_desc}_drift"
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

    drift_cfg = config["drift"]
    round_window_size = args.round_window_size or drift_cfg["round_window_size"]

    # Reference: chronologically LAST N BENIGN sequences from TRAIN (real
    # clients only, matching the centralized-scope convention elsewhere).
    cols_needed = ["temporal_split", "sequence_label", "sequence_index", "window_start_time", client_id_col]
    metadata = pd.read_parquet(seq_dir / "metadata.parquet", columns=cols_needed)
    train_benign = metadata[
        (metadata["temporal_split"] == "train")
        & (metadata[client_id_col] != -1)
        & (metadata["sequence_label"] == "BENIGN")
    ].sort_values("window_start_time")
    n_reference = drift_cfg["psi_reference_window_size"]
    reference_rows = train_benign.tail(n_reference)
    if len(reference_rows) < n_reference:
        logger.warning(
            "Only %d BENIGN train sequences available, fewer than the configured reference size %d -- "
            "using all of them.", len(reference_rows), n_reference,
        )
    reference_indices = reference_rows["sequence_index"].to_numpy()

    # Monitored stream: every TEST sequence (any label), true chronological order.
    test_rows = metadata[
        (metadata["temporal_split"] == "test") & (metadata[client_id_col] != -1)
    ].sort_values("window_start_time")
    stream_indices = test_rows["sequence_index"].to_numpy()
    stream_labels = test_rows["sequence_label"].to_numpy()

    logger.info(
        "Drift monitoring: scope=%s reference=%d benign train sequences, stream=%d test sequences",
        scope_desc, len(reference_indices), len(stream_indices),
    )

    batch_size = config["model"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    reference_loader = DataLoader(
        SequenceIndexDataset(seq_dir, reference_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    stream_loader = DataLoader(
        SequenceIndexDataset(seq_dir, stream_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    device = select_device()
    model_cfg = config["model"]
    window_size = config["sequence"]["window_size"]
    num_features = next(iter(reference_loader))[0].shape[-1]
    model = make_model(num_features, num_classes=2, model_cfg=model_cfg, window_size=window_size).to(device)
    load_shared_checkpoint(best_shared_ckpt, model)

    reference_errors, reference_latents = compute_per_sequence_errors_and_latents(model, reference_loader, device)
    stream_errors, stream_latents = compute_per_sequence_errors_and_latents(model, stream_loader, device)
    logger.info(
        "reference reconstruction MSE: mean=%.4f std=%.4f | stream: mean=%.4f std=%.4f",
        reference_errors.mean(), reference_errors.std(), stream_errors.mean(), stream_errors.std(),
    )

    result = evaluate_drift_stream(
        errors=stream_errors, latents=stream_latents, reference_latents=reference_latents,
        round_window_size=round_window_size, adwin_delta=drift_cfg["adwin_delta"],
        psi_threshold=drift_cfg["psi_threshold"],
        consecutive_rounds_for_retrain=drift_cfg["retrain_trigger_consecutive_rounds"],
    )

    # Label composition per round (reporting only -- never used for detection).
    for report in result["round_reports"]:
        labels_in_round = stream_labels[report["start"]:report["end"]]
        unique, counts = np.unique(labels_in_round, return_counts=True)
        report["label_composition"] = dict(zip(unique.tolist(), counts.tolist()))

    summary = {
        "run_name": run_name,
        "personalized_run_name": personalized_run_name,
        "run_config": {
            "dataset": args.dataset, "alpha": args.alpha, "scheme": scope_desc,
            "round_window_size": round_window_size, "adwin_delta": drift_cfg["adwin_delta"],
            "psi_threshold": drift_cfg["psi_threshold"],
            "consecutive_rounds_for_retrain": drift_cfg["retrain_trigger_consecutive_rounds"],
            "num_reference_sequences": int(len(reference_indices)), "num_stream_sequences": int(len(stream_indices)),
        },
        "reference_reconstruction_mse": {"mean": float(reference_errors.mean()), "std": float(reference_errors.std())},
        "stream_reconstruction_mse": {"mean": float(stream_errors.mean()), "std": float(stream_errors.std())},
        "num_adwin_drift_points": len(result["adwin_drift_indices"]),
        "adwin_drift_indices": result["adwin_drift_indices"],
        "first_trigger_round": result["first_trigger_round"],
        "num_rounds": result["num_rounds"],
        "round_reports": result["round_reports"],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved results to %s", results_path)
    print(f"DONE: {run_name}")
    print(f"  reference={len(reference_indices)} stream={len(stream_indices)} rounds={result['num_rounds']}")
    print(f"  ADWIN drift points: {len(result['adwin_drift_indices'])}")
    print(f"  first retrain-trigger round: {result['first_trigger_round']}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
