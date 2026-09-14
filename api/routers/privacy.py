"""Privacy dashboard: DP configuration, SecAgg+ state, the E3
privacy-utility curve, and the E6 attack evaluation.

The gradient-inversion figures deliberately carry a `diverged` count
alongside the mean: runs where L-BFGS never converged are excluded from
the average and reported separately, because folding a 1e16-scale
non-convergence into a mean would misrepresent it as a reconstruction
error rather than an optimization failure.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.config import EPSILON_VALUES, resolve_scope
from api.repository import (
    rare_labels_for,
    repository,
    safe_float,
    summarize_per_client,
)
from api.routers.health import scope_info
from api.schemas import PrivacyResponse

router = APIRouter(prefix="/privacy", tags=["privacy"])


def _utility_curve(resolved, rare_labels: list[str]) -> list[dict]:
    curve: list[dict] = []
    for eps in EPSILON_VALUES:
        run = repository.load_dp_sweep(resolved, eps)
        if run is None:
            curve.append({
                "epsilon": eps,
                "status": "pending",
                "reason": f"no completed DP run at epsilon={eps} for this scope",
            })
            continue
        summary = summarize_per_client(run["per_client_test_metrics"], rare_labels)
        curve.append({
            "epsilon": eps,
            "achieved_epsilon": safe_float(run.get("achieved_epsilon")),
            "macro_f1": summary["macro_f1"],
            "accuracy": summary["accuracy"],
            "rare_class_recall": summary["rare_class_recall"],
            "num_clients": summary["num_clients"],
            "provenance": "existing",
        })
    return curve


@router.get("", response_model=PrivacyResponse)
def privacy(
    dataset: str = Query(...),
    scope: str | None = Query(None),
) -> PrivacyResponse:
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rare_labels = rare_labels_for(dataset)

    # --- Differential privacy -------------------------------------
    dp_reference = None
    for eps in ("3", "8", "1", "0.5"):
        dp_reference = repository.load_dp_sweep(resolved, eps)
        if dp_reference is not None:
            break

    if dp_reference is not None:
        cfg = dp_reference.get("run_config", {})
        clip_history = dp_reference.get("clip_norm_history", []) or []
        dp_block = {
            "enabled": True,
            "mechanism": "client-level DP (not per-example DP-SGD)",
            "clipping": "adaptive median of this round's client update norms",
            "noise_mechanism": "Gaussian",
            "accountant": "RDP (Opacus)",
            "target_epsilon": safe_float(cfg.get("target_epsilon")),
            "target_delta": safe_float(cfg.get("target_delta")),
            "achieved_epsilon": safe_float(dp_reference.get("achieved_epsilon")),
            "noise_multiplier": safe_float(cfg.get("noise_multiplier")),
            "sample_rate": safe_float(cfg.get("sample_rate")),
            "median_clip_norm": (
                safe_float(sorted(clip_history)[len(clip_history) // 2]) if clip_history else None
            ),
            "sweep_values": EPSILON_VALUES,
            "provenance": "existing",
        }
    else:
        dp_block = {
            "enabled": False,
            "status": "pending",
            "reason": f"no completed DP run for {resolved.label}",
        }

    # --- Secure aggregation ---------------------------------------
    secagg_run = repository.load_mechanism(resolved, "secagg")
    dp_secagg_run = repository.load_mechanism(resolved, "dp_secagg")
    secagg_block: dict = {
        "secagg_only_available": secagg_run is not None,
        "dp_secagg_available": dp_secagg_run is not None,
        "framework": "Flower SecAgg+",
        "provenance": "existing",
    }
    if secagg_run is not None:
        summary = summarize_per_client(secagg_run["per_client_test_metrics"], rare_labels)
        cfg = secagg_run.get("run_config", {})
        secagg_block["secagg_only"] = {
            "macro_f1": summary["macro_f1"],
            "accuracy": summary["accuracy"],
            "clipping_range": cfg.get("clipping_range"),
            "modulus_range": cfg.get("modulus_range"),
            "max_weight": cfg.get("max_weight"),
        }
    if dp_secagg_run is not None:
        summary = summarize_per_client(dp_secagg_run["per_client_test_metrics"], rare_labels)
        cfg = dp_secagg_run.get("run_config", {})
        secagg_block["dp_secagg"] = {
            "macro_f1": summary["macro_f1"],
            "accuracy": summary["accuracy"],
            "clip_bound": safe_float(cfg.get("clip_bound")),
            "noise_multiplier": safe_float(cfg.get("noise_multiplier")),
            "achieved_epsilon": safe_float(dp_secagg_run.get("achieved_epsilon")),
            "note": (
                "Fixed public clip bound applied client-side so SecAgg+ never needs to reveal an "
                "individual update norm -- the combination Phase 8/9 originally could not support."
            ),
        }

    # --- Attack evaluation (E6) -----------------------------------
    tables = repository.load_consolidated_tables() or {}
    e6 = tables.get("E6_empirical_privacy", {})
    scope_key = "cicids2017_alpha5"
    attacks: dict = {}
    if scope_key in e6 and resolved.key == "cicids2017:5":
        attacks = {
            "available": True,
            "scope_note": "Attack evaluation was run for CICIDS2017 alpha=5 only (cost-scoping).",
            "membership_inference": e6[scope_key].get("mia", {}),
            "gradient_inversion": e6[scope_key].get("gradient_inversion", {}),
            "reading_guide": {
                "mia": "AUC 0.5 is chance. Higher means the attack distinguishes members better.",
                "gradient_inversion": (
                    "Reconstruction MSE is a privacy-RISK score: LOWER means MORE leaked. "
                    "Runs that never converged are counted as diverged and excluded from the mean."
                ),
            },
            "provenance": "new",
        }
    else:
        attacks = {
            "available": False,
            "status": "pending",
            "reason": (
                "Gradient inversion and the three-comparator MIA were run for CICIDS2017 alpha=5 only, "
                "matching this project's cost-scoping convention for expensive per-mechanism attacks."
            ),
        }
        mia_own = repository.load_mia(repository.run_name(resolved, "personalized"), "best")
        if mia_own is not None:
            attacks["membership_inference_this_scope"] = {
                "non_dp": mia_own.get("pooled_result", {}),
                "note": "Loss-threshold MIA against this scope's own non-DP personalized checkpoint.",
            }

    return PrivacyResponse(
        scope=scope_info(dataset, scope),
        differential_privacy=dp_block,
        secure_aggregation=secagg_block,
        utility_curve=_utility_curve(resolved, rare_labels),
        attacks=attacks,
    )


@router.get("/utility")
def utility(
    dataset: str = Query(...),
    scope: str | None = Query(None),
) -> dict:
    """The E3 privacy-utility curve on its own, for the chart component."""
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "scope": scope_info(dataset, scope).model_dump(),
        "curve": _utility_curve(resolved, rare_labels_for(dataset)),
        "note": (
            "Rare-class recall reaching exactly 0.000 at every finite epsilon is a real measured "
            "result, not a missing value -- DP eliminates it entirely on this data."
        ),
    }
