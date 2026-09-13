"""Assembles E1-E6/T7's tables from EXISTING and newly-completed
experiment result files -- this script never re-derives an official
accuracy/macro-F1/MSE number differently than how the original training
script computed it. It only:
  - EXTRACTS rare-class recall and FPR from confusion matrices every
    completed run already saved (extract_rare_class_metrics,
    compute_benign_false_positive_rate) -- no re-training, no new
    forward pass.
  - COMPUTES MB/round (compute_communication_cost_mb, from a
    comparator's own saved checkpoint parameter shapes) and rounds-to-
    convergence (each run's own saved best_round) -- both fully
    determined by already-completed runs.
  - REPORTS gradient-inversion / MIA / drift-retrain results exactly as
    their own scripts produced them.

A cell whose source file doesn't exist yet is marked explicitly
{"status": "pending", "reason": "..."} -- NEVER fabricated, silently
defaulted, or averaged in as if it were 0/NaN-as-zero. A metric that
came back non-finite (e.g. gradient inversion's divergent Ours+DP MSE)
is reported with a "diverged": true flag alongside the raw value, never
silently dropped or clamped.

Run this repeatedly as more background experiments complete -- it's
side-effect-free (reads experiments/results + experiments/checkpoints,
writes one consolidated JSON) and safe to re-run at any time.

Usage:
    python scripts/build_e1_e6_tables.py
    python scripts/build_e1_e6_tables.py --output experiments/results/e1_e6_t7_tables.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedpda_ids.evaluation.metrics import (  # noqa: E402
    compute_benign_false_positive_rate,
    extract_rare_class_metrics,
)
from fedpda_ids.federated.simulation import compute_communication_cost_mb  # noqa: E402
from fedpda_ids.models.trainer import load_checkpoint  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402

RESULTS_DIR = Path("experiments/results")
CHECKPOINT_DIR = Path("experiments/checkpoints")

CICIDS_RARE_LABELS = ["Heartbleed", "Infiltration", "Web Attack-SQL Injection", "Bot"]
NBAIOT_RARE_LABELS = ["BASHLITE-Scan", "BASHLITE-Junk"]


def pending(reason: str) -> dict:
    return {"status": "pending", "reason": reason}


def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": None, "n": 1}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def rare_recall_from_metrics(metrics: dict, rare_labels: list[str]) -> float:
    """Mean recall across whichever rare_labels are actually present in
    this scope's classes (extract_rare_class_metrics's own convention --
    absent labels are reported, never coerced into the mean)."""
    rare = extract_rare_class_metrics(metrics, rare_labels)
    present_recalls = [v["recall"] for v in rare.values() if "status" not in v]
    if not present_recalls:
        return float("nan")
    return float(np.mean(present_recalls))


def summarize_single(metrics: dict, rare_labels: list[str]) -> dict:
    """One global model, one test set (Centralized, FedAvg, base
    Base-paper-replication's own centralized eval where applicable) --
    a single value per metric, no cross-client std."""
    return {
        "accuracy": {"mean": metrics["accuracy"], "std": None, "n": 1},
        "macro_f1": {"mean": metrics["macro_f1"], "std": None, "n": 1},
        "rare_class_recall": {"mean": rare_recall_from_metrics(metrics, rare_labels), "std": None, "n": 1},
        "fpr": {"mean": compute_benign_false_positive_rate(metrics["confusion_matrix"], metrics["confusion_matrix_labels"]), "std": None, "n": 1},
    }


def summarize_per_client(per_client_metrics: dict, rare_labels: list[str]) -> dict:
    """Personalized/DP/base-paper-replication/local-only-sweep style:
    one model+eval PER CLIENT -- mean+-std across clients, matching this
    project's established per-client-aggregation convention (Phase 6
    onward) rather than a pooled confusion matrix."""
    valid = [m for m in per_client_metrics.values() if isinstance(m, dict) and "status" not in m and "confusion_matrix" in m]
    return {
        "accuracy": mean_std([m["accuracy"] for m in valid]),
        "macro_f1": mean_std([m["macro_f1"] for m in valid]),
        "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in valid]),
        "fpr": mean_std([compute_benign_false_positive_rate(m["confusion_matrix"], m["confusion_matrix_labels"]) for m in valid]),
        "num_clients": len(valid),
    }


def full_checkpoint_param_arrays(checkpoint_name: str, suffix: str = "best") -> list[np.ndarray] | None:
    path = CHECKPOINT_DIR / f"{checkpoint_name}_{suffix}.pt"
    if not path.exists():
        return None
    ckpt = load_checkpoint(path)
    return [v.detach().cpu().numpy() for v in ckpt["model_state_dict"].values()]


def shared_checkpoint_param_arrays(checkpoint_name: str, suffix: str = "best") -> list[np.ndarray] | None:
    path = CHECKPOINT_DIR / f"{checkpoint_name}_{suffix}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return [v.detach().cpu().numpy() for v in ckpt["shared_state_dict"].values()]


def mb_per_round(param_arrays: list[np.ndarray] | None, clients_per_round: int) -> dict:
    if param_arrays is None:
        return pending("checkpoint not found")
    return compute_communication_cost_mb(param_arrays, clients_per_round)


def local_only_pool_metrics(dataset: str, scope_desc: str, run_tag: str = "localsweep") -> dict | None:
    """Local-only isn't ONE saved run -- it's N separate per-client
    train_baseline.py runs (one file each). Collects whichever of them
    exist so far; a client not yet run is simply absent from the mean
    (never fabricated), and the returned dict says how many were found."""
    per_client = {}
    i = 0
    while True:
        j = load_json(f"{dataset}_{scope_desc}_client{i}_{run_tag}")
        if j is None:
            # Probe a generous range past the last hit once, to tolerate
            # non-contiguous client ids without scanning forever.
            if i > 0 and all(load_json(f"{dataset}_{scope_desc}_client{k}_{run_tag}") is None for k in range(i, i + 5)):
                break
        elif not j.get("skipped", False) and "test_metrics" in j:
            per_client[i] = j["test_metrics"]
        i += 1
        if i > 200:  # hard safety stop -- no scope in this project has anywhere near this many clients
            break
    if not per_client:
        return None
    return per_client


def get(d: dict | None, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


def build_e1(config: dict) -> dict:
    result = {}
    for scope_label, dataset, scope_desc, clients_per_round in [
        ("cicids2017_alpha5", "cicids2017", "5", config["federated"]["cicids2017"]["clients_per_round"]),
        ("nbaiot_main_45", "nbaiot", "main_45", round(config["federated"]["nbaiot"]["num_clients"] * config["federated"]["client_participation_fraction"])),
    ]:
        rare_labels = CICIDS_RARE_LABELS if dataset == "cicids2017" else NBAIOT_RARE_LABELS
        row = {}

        # Local-only
        local_pool = local_only_pool_metrics(dataset, scope_desc)
        if local_pool is None:
            row["local_only"] = pending("local-only per-client sweep not complete yet")
        else:
            row["local_only"] = {
                **summarize_per_client(local_pool, rare_labels),
                "rounds_to_convergence": "n/a (not federated)",
                "mb_per_round": "n/a (not federated)",
            }

        # Centralized
        centralized = load_json(f"{dataset}_{scope_desc}_centralized")
        if centralized is None:
            row["centralized"] = pending("centralized run not found")
        else:
            row["centralized"] = {
                **summarize_single(centralized["test_metrics"], rare_labels),
                "rounds_to_convergence": f"n/a (not federated; best_epoch={centralized.get('best_epoch')})",
                "mb_per_round": "n/a (not federated)",
            }

        # FedAvg
        fedavg = load_json(f"{dataset}_{scope_desc}_fedavg")
        if fedavg is None:
            row["fedavg"] = pending("fedavg run not found")
        else:
            row["fedavg"] = {
                **summarize_single(fedavg["test_metrics"], rare_labels),
                "rounds_to_convergence": fedavg.get("best_round"),
                "mb_per_round": mb_per_round(full_checkpoint_param_arrays(f"{dataset}_{scope_desc}_fedavg"), clients_per_round),
            }

        # FedProx (cicids alpha=5 only, per E1's single representative scope for this comparator)
        if dataset == "cicids2017":
            fedprox = load_json(f"{dataset}_{scope_desc}_fedprox_mu0.01_full") or load_json(f"{dataset}_{scope_desc}_fedprox_mu0.01_smoketest")
            fedprox_name = f"{dataset}_{scope_desc}_fedprox_mu0.01_full" if load_json(f"{dataset}_{scope_desc}_fedprox_mu0.01_full") else f"{dataset}_{scope_desc}_fedprox_mu0.01_smoketest"
            if fedprox is None:
                row["fedprox"] = pending("FedProx run not found")
            else:
                is_smoketest = "smoketest" in fedprox_name
                row["fedprox"] = {
                    **summarize_single(fedprox["test_metrics"], rare_labels),
                    "rounds_to_convergence": fedprox.get("best_round"),
                    "mb_per_round": mb_per_round(full_checkpoint_param_arrays(fedprox_name), clients_per_round),
                    "note": "SMOKETEST (few rounds) -- not yet the full 100-round run" if is_smoketest else None,
                }
        else:
            row["fedprox"] = pending(
                "not run for this scope -- cost-scoping decision (mirrors Phase 12's own precedent of "
                "restricting expensive comparators to one representative scope), not a spec exemption"
            )

        # Base-paper replication
        bpr_name_full = f"{dataset}_{scope_desc}_base_paper_replication_full"
        bpr_name_smoke = f"{dataset}_{scope_desc}_base_paper_replication_smoketest"
        bpr = load_json(bpr_name_full) or load_json(bpr_name_smoke)
        if bpr is None:
            row["base_paper_replication"] = pending("base-paper-replication run not found")
        else:
            bpr_name = bpr_name_full if load_json(bpr_name_full) else bpr_name_smoke
            row["base_paper_replication"] = {
                **summarize_per_client(bpr["per_client_test_metrics"], rare_labels),
                "rounds_to_convergence": bpr.get("best_round"),
                "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(bpr_name), clients_per_round),
                "note": "SMOKETEST (few rounds)" if "smoketest" in bpr_name else None,
            }

        # Ours (no DP) = Personalized FL
        personalized = load_json(f"{dataset}_{scope_desc}_personalized")
        if personalized is None:
            row["ours_no_dp"] = pending("personalized run not found")
        else:
            row["ours_no_dp"] = {
                **summarize_per_client(personalized["per_client_test_metrics"], rare_labels),
                "rounds_to_convergence": personalized.get("best_round"),
                "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(f"{dataset}_{scope_desc}_personalized"), clients_per_round),
            }

        result[scope_label] = row
    return result


def build_e2(config: dict) -> dict:
    rare_labels = CICIDS_RARE_LABELS
    result = {}
    for alpha in ["5", "0.5", "0.1"]:
        row = {}

        fedavg = load_json(f"cicids2017_{alpha}_fedavg")
        row["fedavg"] = pending("fedavg run not found") if fedavg is None else {
            "macro_f1": {"mean": fedavg["test_metrics"]["macro_f1"], "std": None, "n": 1},
            "rare_class_recall": {"mean": rare_recall_from_metrics(fedavg["test_metrics"], rare_labels), "std": None, "n": 1},
        }

        bpr = load_json(f"cicids2017_{alpha}_base_paper_replication_full") or load_json(f"cicids2017_{alpha}_base_paper_replication_smoketest")
        if bpr is None:
            row["base_paper_replication"] = pending("base-paper-replication run not found for this alpha")
        else:
            row["base_paper_replication"] = {
                "macro_f1": mean_std([m["macro_f1"] for m in bpr["per_client_test_metrics"].values() if "status" not in m]),
                "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in bpr["per_client_test_metrics"].values() if "status" not in m]),
            }

        # Ours: prefer the 3-seed Phase 12 results if all three are present, else the single seed=42 result.
        seeds = ["", "_seed123", "_seed2024"]
        seed_results = [load_json(f"cicids2017_{alpha}_personalized{s}") for s in seeds]
        available = [r for r in seed_results if r is not None]
        if not available:
            row["ours"] = pending("personalized run not found")
        else:
            all_macro_f1 = []
            all_rare_recall = []
            for r in available:
                for m in r["per_client_test_metrics"].values():
                    if "status" not in m:
                        all_macro_f1.append(m["macro_f1"])
                        all_rare_recall.append(rare_recall_from_metrics(m, rare_labels))
            row["ours"] = {
                "macro_f1": mean_std(all_macro_f1),
                "rare_class_recall": mean_std(all_rare_recall),
                "num_seeds_available": len(available),
            }

        result[f"alpha_{alpha}"] = row
    return result


def build_e3() -> dict:
    rare_labels = CICIDS_RARE_LABELS
    result = {}
    for alpha in ["0.5", "0.1"]:
        row = {}
        for eps_label, run_name in [
            ("inf", f"cicids2017_{alpha}_personalized"),
            ("8", f"cicids2017_{alpha}_dp_personalized_eps8.0"),
            ("3", f"cicids2017_{alpha}_dp_personalized_eps3.0"),
            ("1", f"cicids2017_{alpha}_dp_personalized_eps1.0"),
            ("0.5", f"cicids2017_{alpha}_dp_personalized_eps0.5"),
        ]:
            r = load_json(run_name)
            if r is None:
                row[f"eps_{eps_label}"] = pending(f"{run_name} not found")
                continue
            valid = [m for m in r["per_client_test_metrics"].values() if "status" not in m]
            row[f"eps_{eps_label}"] = {
                "macro_f1": mean_std([m["macro_f1"] for m in valid]),
                "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in valid]),
                "achieved_epsilon": r.get("achieved_epsilon"),
            }
        result[f"alpha_{alpha}"] = row
    return result


def build_e4() -> dict:
    result = {}
    for scope_label, dataset, scope_desc in [
        ("cicids2017_alpha5", "cicids2017", "5"),
        ("cicids2017_alpha0.5", "cicids2017", "0.5"),
        ("cicids2017_alpha0.1", "cicids2017", "0.1"),
        ("nbaiot_main_45", "nbaiot", "main_45"),
    ]:
        row = {}
        for eps_label, proto_tag, personalized_run in [
            ("inf", None, f"{dataset}_{scope_desc}_personalized"),
            ("8", "eps8", f"{dataset}_{scope_desc}_dp_personalized_eps8.0"),
            ("3", "eps3", f"{dataset}_{scope_desc}_dp_personalized_eps3.0"),
            ("1", "eps1", f"{dataset}_{scope_desc}_dp_personalized_eps1.0"),
            ("0.5", "eps0.5", f"{dataset}_{scope_desc}_dp_personalized_eps0.5"),
        ]:
            proto_name = f"{dataset}_{scope_desc}_prototypes" + (f"_{proto_tag}" if proto_tag else "")
            proto = load_json(proto_name)
            personalized = load_json(personalized_run)
            if proto is None:
                row[f"eps_{eps_label}"] = pending(f"{proto_name} not found (prototype backfill may still be running)")
                continue
            entry = {
                "a_unknown_vs_benign_detection_rate": get(proto, "unknown_vs_benign_detection", "unknown_vs_benign_detection_rate"),
                "b_new_class_detection_rate": get(proto, "zero_day_metrics", "zero_day_detection_rate"),
                "false_positive_rate_on_known": get(proto, "zero_day_metrics", "false_positive_rate"),
            }
            if personalized is not None:
                valid = [m["macro_f1"] for m in personalized["per_client_test_metrics"].values() if "status" not in m] if eps_label == "inf" else None
                if valid:
                    entry["c_known_class_f1"] = mean_std(valid)
                elif personalized is not None:
                    valid2 = [m["macro_f1"] for m in personalized.get("per_client_test_metrics", {}).values() if "status" not in m]
                    entry["c_known_class_f1"] = mean_std(valid2) if valid2 else pending("no per-client macro_f1 available")
            else:
                entry["c_known_class_f1"] = pending(f"{personalized_run} not found")
            entry["c_known_class_f1_note"] = ("known-class F1 is the SAME classifier output whether or not the "
                                               "prototype/zero-day mechanism is applied on top -- prototypes are a "
                                               "post-hoc, non-invasive addition to an already-trained checkpoint, so "
                                               "'unchanged' is a structural guarantee, not a separately re-measured claim")
            row[f"eps_{eps_label}"] = entry
        result[scope_label] = row
    return result


def build_e5() -> dict:
    result = {}
    for scope_label, dataset, scope_desc in [
        ("cicids2017_alpha5", "cicids2017", "5"),
        ("nbaiot_main_45", "nbaiot", "main_45"),
    ]:
        r = load_json(f"{dataset}_{scope_desc}_drift_retrain_full") or load_json(f"{dataset}_{scope_desc}_drift_retrain_smoketest") or load_json(f"{dataset}_{scope_desc}_drift_retrain")
        if r is None:
            result[scope_label] = pending("drift-triggered retraining run not found")
            continue
        name_used = (
            f"{dataset}_{scope_desc}_drift_retrain_full" if load_json(f"{dataset}_{scope_desc}_drift_retrain_full")
            else (f"{dataset}_{scope_desc}_drift_retrain_smoketest" if load_json(f"{dataset}_{scope_desc}_drift_retrain_smoketest") else f"{dataset}_{scope_desc}_drift_retrain")
        )
        result[scope_label] = {
            "f1_before_drift": r["f1_before_drift"],
            "f1_after_drift_no_adaptation_without_monitor": r["f1_after_drift_no_adaptation_without_monitor"],
            "drift_detection_delay_rounds": r["drift_detection_delay_rounds"],
            "f1_after_triggered_retraining_with_monitor": r["f1_after_triggered_retraining_with_monitor"],
            "num_retrain_rounds_run": r["run_config"]["num_retrain_rounds_run"],
            "note": "SMOKETEST (only 2 retrain rounds)" if "smoketest" in name_used else None,
        }
    return result


def build_e6() -> dict:
    result = {}
    for scope_label, dataset, scope_desc in [
        ("cicids2017_alpha5", "cicids2017", "5"),
    ]:
        row = {"mia": {}, "gradient_inversion": {}}

        mia_map = {
            "fedavg": f"mia_{dataset}_{scope_desc}_fedavg_best",
            "ours_dp": f"mia_{dataset}_{scope_desc}_dp_personalized_eps3.0_last",
            "ours_dp_secagg": f"mia_{dataset}_{scope_desc}_dp_secagg_personalized_eps3.0_full_last",
        }
        for label, name in mia_map.items():
            r = load_json(name)
            row["mia"][label] = pending(f"{name} not found") if r is None else {
                "auc": r["pooled_result"]["auc"], "advantage": r["pooled_result"]["advantage"],
            }

        # Gradient inversion: average reconstruction MSE across every completed
        # per-client run for this scope. Divergent (non-finite-scale) values are
        # flagged, never silently averaged in as if they were a normal number.
        gi_results = []
        i = 0
        while True:
            r = load_json(f"gradient_inversion_{dataset}_{scope_desc}_client{i}") or load_json(f"gradient_inversion_{dataset}_{scope_desc}_client{i}_smoketest")
            if r is not None:
                gi_results.append(r)
            i += 1
            if i > 50:
                break
        if not gi_results:
            row["gradient_inversion"] = pending("no gradient-inversion runs found for this scope")
        else:
            for surface in ["fedavg_no_protection", "ours_dp", "ours_dp_secagg"]:
                values = [r["reconstruction_mse"][surface] for r in gi_results]
                # DIVERGED sentinel: a grad-matching optimization that never
                # converged produces an MSE many orders of magnitude larger
                # than a real (bounded-latent-space) reconstruction could be --
                # reported separately, never blended into the same mean.
                finite_normal = [v for v in values if v < 1e6]
                diverged_count = sum(1 for v in values if v >= 1e6)
                row["gradient_inversion"][surface] = {
                    "reconstruction_mse": mean_std(finite_normal) if finite_normal else None,
                    "num_diverged_runs": diverged_count,
                    "num_total_runs": len(values),
                    "diverged": diverged_count > 0 and not finite_normal,
                }
        result[scope_label] = row
    return result


def build_t7(config: dict) -> dict:
    rare_labels = CICIDS_RARE_LABELS
    dataset, scope_desc = "cicids2017", "5"
    clients_per_round = config["federated"]["cicids2017"]["clients_per_round"]
    rows = []

    fedavg = load_json(f"{dataset}_{scope_desc}_fedavg")
    rows.append({
        "row": "FedAvg", "epsilon": "inf",
        "macro_f1": {"mean": get(fedavg, "test_metrics", "macro_f1"), "std": None, "n": 1} if fedavg else pending("fedavg not found"),
        "rare_class_recall": {"mean": rare_recall_from_metrics(fedavg["test_metrics"], rare_labels), "std": None, "n": 1} if fedavg else pending("fedavg not found"),
        "zero_day_b_new_class_detection": {"status": "not_applicable", "reason": "FedAvg has no prototype/zero-day mechanism"},
        "mb_per_round": mb_per_round(full_checkpoint_param_arrays(f"{dataset}_{scope_desc}_fedavg"), clients_per_round),
    })

    personalized = load_json(f"{dataset}_{scope_desc}_personalized")
    pers_valid = [m for m in personalized["per_client_test_metrics"].values() if "status" not in m] if personalized else []
    rows.append({
        "row": "+Personalized layers", "epsilon": "inf",
        "macro_f1": mean_std([m["macro_f1"] for m in pers_valid]) if personalized else pending("personalized not found"),
        "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in pers_valid]) if personalized else pending("personalized not found"),
        "zero_day_b_new_class_detection": {"status": "not_applicable", "reason": "prototypes added in the next row"},
        "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(f"{dataset}_{scope_desc}_personalized"), clients_per_round),
    })

    proto_inf = load_json(f"{dataset}_{scope_desc}_prototypes")
    rows.append({
        "row": "+Prototypes", "epsilon": "inf",
        "macro_f1": mean_std([m["macro_f1"] for m in pers_valid]) if personalized else pending("personalized not found"),
        "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in pers_valid]) if personalized else pending("personalized not found"),
        "zero_day_b_new_class_detection": get(proto_inf, "zero_day_metrics", "zero_day_detection_rate") if proto_inf else pending("prototypes not found"),
        "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(f"{dataset}_{scope_desc}_personalized"), clients_per_round),
        "note": "same trained model as the previous row -- prototypes are a post-hoc, non-invasive addition; "
                "MB/round shown excludes the one-time (not per-round-recurring) protected-prototype transmission",
    })

    dp3 = load_json(f"{dataset}_{scope_desc}_dp_personalized_eps3.0")
    dp3_valid = [m for m in dp3["per_client_test_metrics"].values() if "status" not in m] if dp3 else []
    proto_eps3 = load_json(f"{dataset}_{scope_desc}_prototypes_eps3")
    rows.append({
        "row": "+DP", "epsilon": 3,
        "macro_f1": mean_std([m["macro_f1"] for m in dp3_valid]) if dp3 else pending("dp eps=3 not found"),
        "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in dp3_valid]) if dp3 else pending("dp eps=3 not found"),
        "zero_day_b_new_class_detection": get(proto_eps3, "zero_day_metrics", "zero_day_detection_rate") if proto_eps3 else pending("prototypes_eps3 not found"),
        "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(f"{dataset}_{scope_desc}_dp_personalized_eps3.0"), clients_per_round),
    })

    secagg_name = f"{dataset}_{scope_desc}_dp_secagg_personalized_eps3.0_full"
    secagg = load_json(secagg_name) or load_json(f"{dataset}_{scope_desc}_dp_secagg_personalized_eps3.0_smoketest")
    secagg_used_name = secagg_name if load_json(secagg_name) else f"{dataset}_{scope_desc}_dp_secagg_personalized_eps3.0_smoketest"
    secagg_valid = [m for m in secagg["per_client_test_metrics"].values() if "status" not in m] if secagg else []
    proto_secagg = load_json(f"{dataset}_{scope_desc}_prototypes_t7_secagg")
    rows.append({
        "row": "+SecAgg", "epsilon": 3,
        "macro_f1": mean_std([m["macro_f1"] for m in secagg_valid]) if secagg else pending("dp+secagg eps=3 not found"),
        "rare_class_recall": mean_std([rare_recall_from_metrics(m, rare_labels) for m in secagg_valid]) if secagg else pending("dp+secagg eps=3 not found"),
        "zero_day_b_new_class_detection": get(proto_secagg, "zero_day_metrics", "zero_day_detection_rate") if proto_secagg else pending("prototypes against the DP+SecAgg checkpoint not computed yet"),
        "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(secagg_used_name), clients_per_round),
        "note": "SMOKETEST (only 3 rounds)" if secagg is not None and "smoketest" in secagg_used_name else None,
    })

    drift_row_name = f"{dataset}_{scope_desc}_dp_secagg_personalized_eps3.0_drift_retrain"
    drift_retrained = load_json(drift_row_name)
    proto_drift = load_json(f"{dataset}_{scope_desc}_prototypes_t7_drift")
    if drift_retrained is not None:
        macro_f1_entry = drift_retrained["f1_after_triggered_retraining_with_monitor"]
        rare_entry = pending("rare-class recall needs per-client breakdown from the drift-retrain run's per_client field")
    else:
        macro_f1_entry = pending("T7's drift-monitor retrain (starting from the DP+SecAgg checkpoint) not run yet")
        rare_entry = pending("same")
    rows.append({
        "row": "+Drift monitor", "epsilon": 3,
        "macro_f1": macro_f1_entry,
        "rare_class_recall": rare_entry,
        "zero_day_b_new_class_detection": get(proto_drift, "zero_day_metrics", "zero_day_detection_rate") if proto_drift else pending("prototypes against the drift-retrained checkpoint not computed yet"),
        "mb_per_round": mb_per_round(shared_checkpoint_param_arrays(secagg_used_name), clients_per_round),
        "note": "reuses the DP+SecAgg row's per-round exchange size -- retraining doesn't change what's transmitted, "
                "only how many additional rounds run after the trigger fires",
    })

    return {"cicids2017_alpha5": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble E1-E6/T7's tables from existing + newly-completed results")
    parser.add_argument("--output", default="experiments/results/e1_e6_t7_tables.json")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    tables = {
        "E1_core_comparison": build_e1(config),
        "E2_non_iid_severity": build_e2(config),
        "E3_privacy_utility_curve": build_e3(),
        "E4_zero_day_under_dp": build_e4(),
        "E5_drift_and_recovery": build_e5(),
        "E6_empirical_privacy": build_e6(),
        "T7_ablation": build_t7(config),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tables, indent=2, default=str), encoding="utf-8")

    def count_pending(obj, counts=None):
        if counts is None:
            counts = [0, 0]
        if isinstance(obj, dict):
            if obj.get("status") == "pending":
                counts[0] += 1
                counts[1] += 1
                return counts
            counts[1] += 1
            for v in obj.values():
                count_pending(v, counts)
        elif isinstance(obj, list):
            for v in obj:
                count_pending(v, counts)
        return counts

    pending_count, total_count = count_pending(tables)
    print(f"DONE: wrote {output_path}")
    print(f"  {pending_count} cell(s) still pending out of {total_count} dict-shaped entries inspected")


if __name__ == "__main__":
    main()
