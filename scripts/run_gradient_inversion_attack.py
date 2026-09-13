"""E6's gradient inversion attack (DLG) -- the missing half of E6's
Table T6 (Phase 11 already built loss-threshold MIA; this adds the
"reconstruction quality" column), run against exactly the three
comparators the frozen spec names: FedAvg updates / Ours+DP /
Ours+DP+SecAgg. See src/fedpda_ids/privacy/gradient_inversion.py's
module docstring for the full attack mechanism and why each surface is
defined the way it is.

Design choice, not a shortcut: rather than attacking three DIFFERENT
runs' final trained weights (which would confound "how well-converged
was this checkpoint" with "how much does this protection mechanism
leak"), this script computes ONE real gradient for a target client at
a chosen checkpoint, then applies each protection mechanism to THAT
SAME gradient -- isolating the mechanism's effect on leakage, not
differences in training convergence between runs. The FedAvg surface
still uses its own separate (Phase 5) checkpoint, since that's a
genuinely different (unpersonalized, full-model) exchange contract.

Real calibration values (clip_bound, noise_multiplier) are sourced from
an EXISTING completed run's saved run_config/clip_norm_history --
never re-guessed -- so the attack reflects the REAL noise level those
mechanisms actually use in this project's own completed experiments.

Usage:
    python scripts/run_gradient_inversion_attack.py --dataset cicids2017 --alpha 5 \
        --fedavg-run-name cicids2017_5_fedavg \
        --personalized-run-name cicids2017_5_personalized \
        --dp-run-name cicids2017_5_dp_personalized_eps3.0 \
        --dp-secagg-run-name cicids2017_5_dp_secagg_personalized_eps3.0_smoketest \
        --target-client-id 1 --num-iterations 150
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedpda_ids.data.sequence_dataset import build_scope_dataloaders  # noqa: E402
from fedpda_ids.federated.client import SHARED_PREFIXES  # noqa: E402
from fedpda_ids.federated.simulation import build_trainable_client_pool, load_shared_checkpoint, make_model  # noqa: E402
from fedpda_ids.models.trainer import load_checkpoint  # noqa: E402
from fedpda_ids.privacy.dp import add_gaussian_noise, clip_update  # noqa: E402
from fedpda_ids.privacy.gradient_inversion import (  # noqa: E402
    aggregate_gradients,
    compute_client_gradient,
    dlg_reconstruct,
    reconstruction_mse,
)
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import resolve_scope  # noqa: E402


def _load_run_config(results_dir: Path, run_name: str) -> dict:
    path = results_dir / f"{run_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the corresponding training script first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="E6: gradient inversion (DLG) attack -- Table T6's reconstruction-quality column")
    parser.add_argument("--dataset", choices=["cicids2017", "nbaiot"], required=True)
    parser.add_argument("--alpha", choices=["5", "0.5", "0.1"], default=None)
    parser.add_argument("--scheme", choices=["main_45", "ablation_9"], default=None)
    parser.add_argument("--fedavg-run-name", type=str, required=True, help="Phase 5 plain-FedAvg run (full-model checkpoint)")
    parser.add_argument("--personalized-run-name", type=str, required=True, help="Phase 6 non-DP personalized run (source of the real gradient for the Ours+* surfaces)")
    parser.add_argument("--dp-run-name", type=str, required=True, help="a completed DPPersonalizedFedAvg run -- sources real clip_norm/noise_multiplier for the Ours+DP surface")
    parser.add_argument("--dp-secagg-run-name", type=str, required=True, help="a completed DPSecAggPersonalizedFedAvg run -- sources real clip_bound/noise_multiplier/clients_per_round for the Ours+DP+SecAgg surface")
    parser.add_argument("--target-client-id", type=int, required=True)
    parser.add_argument("--attack-batch-size", type=int, default=1,
                         help="DLG is a documented, literature-wide limitation past a handful of examples -- keep this small")
    parser.add_argument("--num-iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"], level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"], run_name="run_gradient_inversion_attack",
    )

    scope_desc = args.alpha if args.dataset == "cicids2017" else (args.scheme or "main_45")
    run_name = f"gradient_inversion_{args.dataset}_{scope_desc}_client{args.target_client_id}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"

    results_dir = Path(config["training"]["results_dir"])
    results_path = results_dir / f"{run_name}.json"
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f"{results_path} exists -- pass --run-tag or --overwrite.")

    _load_run_config(results_dir, args.fedavg_run_name)  # existence check only -- calibration comes from dp/dp_secagg results
    dp_results = _load_run_config(results_dir, args.dp_run_name)
    dp_secagg_results = _load_run_config(results_dir, args.dp_secagg_run_name)

    label_to_index = dp_results["run_config"]["label_to_index"]
    num_classes = len(label_to_index)

    seq_dir, client_id_col, _ = resolve_scope(config, args.dataset, args.alpha, args.scheme)
    model_cfg = config["model"]
    window_size = config["sequence"]["window_size"]
    lambda_ce = model_cfg["loss"]["lambda_ce"]
    fed_cfg = config["federated"]

    centralized_scope = build_scope_dataloaders(seq_dir, client_id_col, None, model_cfg["batch_size"], 0, False, label_to_index)
    num_features = next(iter(centralized_scope["loaders"]["train"]))[0].shape[-1]

    # -----------------------------------------------------------------
    # Real calibration values: sourced from ALREADY-COMPLETED runs, not guessed.
    # -----------------------------------------------------------------
    dp_clip_norm_history = dp_results.get("clip_norm_history", [])
    if not dp_clip_norm_history:
        raise ValueError(f"{args.dp_run_name} has no clip_norm_history -- is this a DPPersonalizedFedAvg run?")
    dp_clip_norm = float(np.median(dp_clip_norm_history))
    dp_noise_multiplier = dp_results["run_config"]["noise_multiplier"]

    dp_secagg_clip_bound = dp_secagg_results["run_config"]["clip_bound"]
    dp_secagg_noise_multiplier = dp_secagg_results["run_config"]["noise_multiplier"]
    dp_secagg_clients_per_round = dp_secagg_results["run_config"]["clients_per_round"]

    logger.info(
        "Real calibration: dp_clip_norm(median)=%.4f dp_noise_multiplier=%.4f | "
        "dp_secagg_clip_bound=%.4f dp_secagg_noise_multiplier=%.4f dp_secagg_m=%d",
        dp_clip_norm, dp_noise_multiplier, dp_secagg_clip_bound, dp_secagg_noise_multiplier, dp_secagg_clients_per_round,
    )

    # -----------------------------------------------------------------
    # Real client pool + target client's real train batch.
    # -----------------------------------------------------------------
    num_clients_configured = fed_cfg["cicids2017"]["num_clients"] if args.dataset == "cicids2017" else fed_cfg["nbaiot"]["num_clients"]
    min_seq = config["training"]["local_training"]["min_train_sequences"]
    min_cls = config["training"]["local_training"]["min_train_classes"]
    pool = build_trainable_client_pool(seq_dir, client_id_col, num_clients_configured, min_seq, min_cls)
    if args.target_client_id not in pool:
        raise ValueError(f"client {args.target_client_id} is not in the trainable pool for this scope.")
    other_client_ids = [c for c in pool if c != args.target_client_id][: dp_secagg_clients_per_round - 1]
    if len(other_client_ids) < dp_secagg_clients_per_round - 1:
        raise ValueError(f"pool too small to fill clients_per_round={dp_secagg_clients_per_round} for the SecAgg+ surface.")

    def real_batch(client_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        scope = build_scope_dataloaders(seq_dir, client_id_col, client_id, args.attack_batch_size, 0, False, label_to_index)
        x, y = next(iter(scope["loaders"]["train"]))
        return x[: args.attack_batch_size], y[: args.attack_batch_size]

    target_x, target_y = real_batch(args.target_client_id)

    # -----------------------------------------------------------------
    # Surface 1: "FedAvg updates" -- full model, no protection.
    # -----------------------------------------------------------------
    fedavg_model = make_model(num_features, num_classes, model_cfg, window_size)
    fedavg_ckpt = load_checkpoint(Path(config["training"]["checkpoint_dir"]) / f"{args.fedavg_run_name}_best.pt")
    fedavg_model.load_state_dict(fedavg_ckpt["model_state_dict"])
    fedavg_grad = compute_client_gradient(fedavg_model, target_x, target_y, lambda_ce)
    fedavg_dlg = dlg_reconstruct(
        fedavg_model, fedavg_grad, target_x.shape, num_classes, lambda_ce,
        num_iterations=args.num_iterations, lr=args.lr, seed=args.seed,
    )
    fedavg_mse = reconstruction_mse(target_x, fedavg_dlg["dummy_x"])
    logger.info("Surface 1/3 (FedAvg, no protection) done: mse=%.6f", fedavg_mse)

    # -----------------------------------------------------------------
    # Surfaces 2 & 3 share the SAME base model + SAME real target gradient
    # (only shared encoder/decoder params -- personalized FL's transmitted subset).
    # -----------------------------------------------------------------
    shared_model = make_model(num_features, num_classes, model_cfg, window_size)
    load_shared_checkpoint(Path(config["training"]["checkpoint_dir"]) / f"{args.personalized_run_name}_best.pt", shared_model)
    target_shared_grad = compute_client_gradient(shared_model, target_x, target_y, lambda_ce, shared_only_prefixes=SHARED_PREFIXES)

    # Surface 2: "Ours+DP" -- clip+noise the SAME real gradient with
    # THIS DP run's own real (adaptive-median) clip norm + noise_multiplier.
    target_grad_np = [g.numpy() for g in target_shared_grad]
    dp_clipped = clip_update(target_grad_np, dp_clip_norm)
    dp_noised = add_gaussian_noise(dp_clipped, dp_noise_multiplier, dp_clip_norm, np.random.default_rng(args.seed))
    dp_grad = [torch.tensor(a) for a in dp_noised]
    dp_dlg = dlg_reconstruct(
        shared_model, dp_grad, target_x.shape, num_classes, lambda_ce, shared_only_prefixes=SHARED_PREFIXES,
        num_iterations=args.num_iterations, lr=args.lr, seed=args.seed,
    )
    dp_mse = reconstruction_mse(target_x, dp_dlg["dummy_x"])
    logger.info("Surface 2/3 (Ours+DP) done: mse=%.6f", dp_mse)

    # Surface 3: "Ours+DP+SecAgg" -- exactly DPSecAggPersonalizedFedAvg's
    # own math: each of m clients clips its own real gradient to the fixed
    # public clip_bound, SecAgg+ reveals their MEAN, then ONE noise draw
    # (sensitivity clip_bound/m) is added to that mean.
    other_grads = []
    for other_id in other_client_ids:
        other_x, other_y = real_batch(other_id)
        other_grad = compute_client_gradient(shared_model, other_x, other_y, lambda_ce, shared_only_prefixes=SHARED_PREFIXES)
        other_grads.append([g.numpy() for g in other_grad])
    clipped_target = clip_update(target_grad_np, dp_secagg_clip_bound)
    clipped_others = [clip_update(g, dp_secagg_clip_bound) for g in other_grads]
    mean_clipped = aggregate_gradients([[torch.tensor(a) for a in clipped_target]] + [[torch.tensor(a) for a in g] for g in clipped_others])
    mean_clipped_np = [t.numpy() for t in mean_clipped]
    noised_mean = add_gaussian_noise(
        mean_clipped_np, dp_secagg_noise_multiplier, dp_secagg_clip_bound / dp_secagg_clients_per_round, np.random.default_rng(args.seed),
    )
    secagg_grad = [torch.tensor(a) for a in noised_mean]
    secagg_dlg = dlg_reconstruct(
        shared_model, secagg_grad, target_x.shape, num_classes, lambda_ce, shared_only_prefixes=SHARED_PREFIXES,
        num_iterations=args.num_iterations, lr=args.lr, seed=args.seed,
    )
    secagg_mse = reconstruction_mse(target_x, secagg_dlg["dummy_x"])
    logger.info("Surface 3/3 (Ours+DP+SecAgg, m=%d) done: mse=%.6f", dp_secagg_clients_per_round, secagg_mse)

    summary = {
        "run_name": run_name,
        "dataset": args.dataset, "alpha": args.alpha, "scheme": args.scheme,
        "target_client_id": args.target_client_id, "other_client_ids": other_client_ids,
        "attack_batch_size": args.attack_batch_size, "num_iterations": args.num_iterations,
        "fedavg_run_name": args.fedavg_run_name, "personalized_run_name": args.personalized_run_name,
        "dp_run_name": args.dp_run_name, "dp_secagg_run_name": args.dp_secagg_run_name,
        "calibration": {
            "dp_clip_norm_median": dp_clip_norm, "dp_noise_multiplier": dp_noise_multiplier,
            "dp_secagg_clip_bound": dp_secagg_clip_bound, "dp_secagg_noise_multiplier": dp_secagg_noise_multiplier,
            "dp_secagg_clients_per_round": dp_secagg_clients_per_round,
        },
        "reconstruction_mse": {
            "fedavg_no_protection": fedavg_mse,
            "ours_dp": dp_mse,
            "ours_dp_secagg": secagg_mse,
        },
        "final_grad_distance": {
            "fedavg_no_protection": fedavg_dlg["final_grad_distance"],
            "ours_dp": dp_dlg["final_grad_distance"],
            "ours_dp_secagg": secagg_dlg["final_grad_distance"],
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved results to %s", results_path)
    print(f"DONE: {run_name}")
    print("  reconstruction MSE (LOWER = more leaked -- this is a privacy-risk score):")
    print(f"    FedAvg updates (no protection): {fedavg_mse:.6f}")
    print(f"    Ours+DP:                        {dp_mse:.6f}")
    print(f"    Ours+DP+SecAgg:                 {secagg_mse:.6f}")
    print(f"  results: {results_path}")


if __name__ == "__main__":
    main()
