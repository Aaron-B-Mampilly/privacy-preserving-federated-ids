"""Federated learning runs.

These are COMPLETED runs replayed from their saved histories -- this
project trains offline in multi-hour batches, so there is no live round
counter to stream. The mechanism list is discovered from what actually
exists on disk for a scope, so a scope that never had (say) a FedProx run
simply doesn't offer one rather than showing an empty shell.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.config import MECHANISM_SUFFIXES, resolve_scope
from api.repository import (
    rare_labels_for,
    repository,
    safe_float,
    summarize_global,
    summarize_per_client,
)
from api.routers.dashboard import _series_from_history
from api.routers.health import scope_info
from api.schemas import ChartSeries, FLRunDetail, FLRunSummary, MetricValue, Provenance

router = APIRouter(prefix="/fl", tags=["federated-learning"])

MECHANISM_LABELS = {
    "centralized": "Centralized (non-federated reference)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx (mu=0.01)",
    "base_paper_replication": "Base-paper replication",
    "personalized": "Ours - Personalized FL",
    "secagg": "Ours + SecAgg+",
    "dp_secagg": "Ours + DP + SecAgg+ (eps=3)",
}

# Mechanisms that exchange the FULL model rather than shared layers only.
FULL_MODEL_MECHANISMS = {"fedavg", "fedprox", "centralized"}


def _mb_per_round_for(scope, mechanism: str) -> float | None:
    """Reads the already-computed MB/round out of the consolidated E1
    table rather than recomputing it, so the value matches the thesis."""
    tables = repository.load_consolidated_tables() or {}
    e1 = tables.get("E1_core_comparison", {})
    scope_key = "cicids2017_alpha5" if scope.key == "cicids2017:5" else (
        "nbaiot_main_45" if scope.dataset == "nbaiot" else None
    )
    if not scope_key or scope_key not in e1:
        return None
    entry = e1[scope_key].get({"personalized": "ours_no_dp"}.get(mechanism, mechanism), {})
    mb = entry.get("mb_per_round") if isinstance(entry, dict) else None
    return safe_float(mb.get("mb_per_round")) if isinstance(mb, dict) else None


@router.get("/runs", response_model=list[FLRunSummary])
def list_runs(
    dataset: str = Query(...),
    scope: str | None = Query(None),
) -> list[FLRunSummary]:
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rare_labels = rare_labels_for(dataset)
    out: list[FLRunSummary] = []

    for mechanism in MECHANISM_SUFFIXES:
        run = repository.load_mechanism(resolved, mechanism)
        if run is None:
            continue
        run_name = repository.run_name(resolved, mechanism)
        cfg = run.get("run_config", {})

        if "per_client_test_metrics" in run:
            summary = summarize_per_client(run["per_client_test_metrics"], rare_labels)
        elif "test_metrics" in run:
            summary = summarize_global(run["test_metrics"], rare_labels)
        else:
            continue

        rounds = cfg.get("num_rounds") or cfg.get("epochs") or 0
        out.append(
            FLRunSummary(
                run_name=run_name,
                mechanism=mechanism,
                label=MECHANISM_LABELS.get(mechanism, mechanism),
                rounds=int(rounds) if rounds else 0,
                best_round=run.get("best_round") if isinstance(run.get("best_round"), int) else None,
                clients_configured=int(cfg.get("num_clients_configured") or resolved.num_clients),
                clients_per_round=int(cfg.get("clients_per_round") or resolved.clients_per_round),
                local_epochs=cfg.get("local_epochs"),
                accuracy=MetricValue(**summary["accuracy"], provenance=Provenance.EXISTING),
                macro_f1=MetricValue(**summary["macro_f1"], provenance=Provenance.EXISTING),
                mb_per_round=_mb_per_round_for(resolved, mechanism),
            )
        )
    return out


@router.get("/runs/{run_name}", response_model=FLRunDetail)
def run_detail(run_name: str) -> FLRunDetail:
    run = repository.load(run_name)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No completed run named {run_name!r}")

    cfg = run.get("run_config", {})
    dataset = "nbaiot" if run_name.startswith("nbaiot") else "cicids2017"
    scope_token = run_name.split("_")[1] if dataset == "cicids2017" else "main_45"
    try:
        resolved = resolve_scope(dataset, scope_token)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"cannot resolve scope from run name: {exc}") from exc

    mechanism = next(
        (m for m, suffix in MECHANISM_SUFFIXES.items() if run_name.endswith(suffix)),
        "unknown",
    )

    distributed = run.get("history_metrics_distributed", {}) or {}
    series: list[ChartSeries] = [
        _series_from_history(distributed.get("macro_f1"), "macro_f1", "Macro-F1 per round"),
        _series_from_history(distributed.get("accuracy"), "accuracy", "Accuracy per round"),
        _series_from_history(run.get("history_losses_centralized"), "val_loss", "Centralized val loss/MSE"),
        _series_from_history(run.get("history_losses_distributed"), "client_loss", "Distributed client loss"),
    ]
    mb = _mb_per_round_for(resolved, mechanism)
    if mb is not None and distributed.get("macro_f1"):
        series.append(
            ChartSeries(
                key="mb_per_round",
                label="Communication MB per round",
                points=[
                    {"round": int(e[0]), "value": mb} for e in distributed["macro_f1"]
                ],
                provenance=Provenance.DERIVED,
                unit="MB",
            )
        )

    # Per-client panel. Only non-identifying aggregates are exposed:
    # a client index, whether it produced an evaluated model, and its own
    # test-set size/score -- never raw client data or feature values.
    clients: list[dict] = []
    per_client = run.get("per_client_test_metrics", {})
    for client_id, metrics in sorted(per_client.items(), key=lambda kv: int(kv[0])):
        if not isinstance(metrics, dict):
            continue
        if "status" in metrics:
            clients.append({
                "client_id": int(client_id),
                "status": metrics["status"],
                "participated": False,
                "num_samples": None,
                "accuracy": None,
                "macro_f1": None,
            })
        else:
            clients.append({
                "client_id": int(client_id),
                "status": "evaluated",
                "participated": True,
                "num_samples": metrics.get("num_samples"),
                "accuracy": safe_float(metrics.get("accuracy")),
                "macro_f1": safe_float(metrics.get("macro_f1")),
            })

    best_round = run.get("best_round") if isinstance(run.get("best_round"), int) else None
    convergence_note = (
        "Best round is selected by lowest centralized validation MSE (the frozen policy). "
        "On N-BaIoT full-model runs this can select a very early round while macro-F1 sits on a "
        "flat, noisy plateau -- reported as-is rather than re-selected retrospectively."
    )

    return FLRunDetail(
        run_name=run_name,
        mechanism=mechanism,
        scope=scope_info(resolved.dataset, resolved.scope),
        config={
            "algorithm": MECHANISM_LABELS.get(mechanism, mechanism),
            "num_rounds": cfg.get("num_rounds"),
            "local_epochs": cfg.get("local_epochs"),
            "batch_size": cfg.get("batch_size"),
            "clients_configured": cfg.get("num_clients_configured"),
            "clients_per_round": cfg.get("clients_per_round"),
            "clients_in_pool": cfg.get("num_clients_pool"),
            "participation_rate": (
                round(cfg["clients_per_round"] / cfg["num_clients_pool"], 4)
                if cfg.get("clients_per_round") and cfg.get("num_clients_pool") else None
            ),
            "personalized": cfg.get("personalized", False),
            "seed": cfg.get("seed"),
            "elapsed_seconds": safe_float(run.get("elapsed_seconds")),
            "strategy": cfg.get("strategy_cls"),
        },
        series=series,
        clients=clients,
        best_round=best_round,
        convergence_note=convergence_note,
    )
