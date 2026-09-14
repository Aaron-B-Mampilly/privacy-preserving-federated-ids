"""Dashboard overview.

KPIs and chart series are assembled from COMPLETED runs' saved
histories. FL training in this project is an hours-long offline batch and
the Prometheus exporter only runs while such a job is executing, so there
is no live trainer to stream from -- the round-by-round series here are a
replay of a finished run's own recorded history, and the response says so
via each series' provenance tag.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.config import CHECKPOINT_DIR, EPSILON_VALUES, resolve_scope
from api.repository import (
    benign_fpr,
    mean_std,
    rare_labels_for,
    rare_recall,
    repository,
    safe_float,
    summarize_global,
    summarize_per_client,
)
from api.routers.health import scope_info
from api.schemas import (
    ChartSeries,
    DashboardResponse,
    EventItem,
    MetricValue,
    Provenance,
    SeriesPoint,
)

router = APIRouter(tags=["dashboard"])


def _series_from_history(history: list, key: str, label: str, unit: str | None = None) -> ChartSeries:
    """Flower history entries are [round, value] pairs."""
    points = []
    for entry in history or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            points.append(SeriesPoint(round=int(entry[0]), value=safe_float(entry[1])))
    return ChartSeries(key=key, label=label, points=points, provenance=Provenance.EXISTING, unit=unit)


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    dataset: str = Query(..., description="cicids2017 | nbaiot"),
    scope: str | None = Query(None, description="alpha for CICIDS2017, scheme for N-BaIoT"),
) -> DashboardResponse:
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rare_labels = rare_labels_for(dataset)
    run = repository.load_mechanism(resolved, "personalized")
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"No completed personalized run for {resolved.label}. Nothing to display yet.",
        )
    run_name = repository.run_name(resolved, "personalized")
    summary = summarize_per_client(run["per_client_test_metrics"], rare_labels)

    # --- KPI cards -------------------------------------------------
    kpis: dict[str, MetricValue] = {
        "macro_f1": MetricValue(**summary["macro_f1"], provenance=Provenance.EXISTING),
        "accuracy": MetricValue(**summary["accuracy"], provenance=Provenance.EXISTING),
        "rare_class_recall": MetricValue(**summary["rare_class_recall"], provenance=Provenance.DERIVED),
        "fpr": MetricValue(**summary["fpr"], provenance=Provenance.DERIVED),
    }

    # Zero-day: (b) NEW CLASS detection rate at eps=inf.
    protos = repository.load_prototypes(resolved, "inf")
    if protos is not None:
        kpis["zero_day_detection"] = MetricValue(
            mean=safe_float(protos.get("zero_day_metrics", {}).get("zero_day_detection_rate")),
            n=1,
            provenance=Provenance.EXISTING,
            note="NEW CLASS detection rate at epsilon=inf",
        )
    else:
        kpis["zero_day_detection"] = MetricValue(note="Prototype results not computed for this scope")

    # Privacy budget: the tightest epsilon with a completed DP run.
    achieved = None
    for eps in ("0.5", "1", "3", "8"):
        dp_run = repository.load_dp_sweep(resolved, eps)
        if dp_run is not None:
            achieved = safe_float(dp_run.get("achieved_epsilon"))
            break
    kpis["privacy_epsilon"] = MetricValue(
        mean=achieved,
        n=1 if achieved is not None else 0,
        provenance=Provenance.EXISTING,
        note="Tightest completed DP budget (achieved epsilon)" if achieved else "No DP run for this scope",
    )

    run_config = run.get("run_config", {})
    total_rounds = run_config.get("num_rounds")
    kpis["current_round"] = MetricValue(
        mean=float(total_rounds) if total_rounds else None,
        n=1 if total_rounds else 0,
        provenance=Provenance.EXISTING,
        note="Final round of the completed run (training is offline, not live)",
    )
    kpis["convergence_round"] = MetricValue(
        mean=safe_float(run.get("best_round")),
        n=1,
        provenance=Provenance.EXISTING,
        note="Best round by centralized validation MSE",
    )
    kpis["active_clients"] = MetricValue(
        mean=float(summary["num_clients"]),
        n=1,
        provenance=Provenance.EXISTING,
        note=f"{summary['num_clients']} of {resolved.num_clients} configured clients evaluated",
    )

    # Communication cost, derived from this run's own checkpoint shapes.
    mb_per_round = None
    tables = repository.load_consolidated_tables() or {}
    e1 = tables.get("E1_core_comparison", {})
    scope_key = "cicids2017_alpha5" if resolved.key == "cicids2017:5" else (
        "nbaiot_main_45" if resolved.dataset == "nbaiot" else None
    )
    if scope_key and scope_key in e1:
        ours = e1[scope_key].get("ours_no_dp", {})
        mb = ours.get("mb_per_round")
        if isinstance(mb, dict):
            mb_per_round = safe_float(mb.get("mb_per_round"))
    kpis["mb_per_round"] = MetricValue(
        mean=mb_per_round,
        n=1 if mb_per_round is not None else 0,
        provenance=Provenance.DERIVED,
        note="Both directions, all sampled clients" if mb_per_round else "Not computed for this scope",
    )

    # Drift status KPI.
    drift = repository.load_drift(resolved)
    if drift is not None:
        trigger = drift.get("first_trigger_round")
        kpis["drift_delay_rounds"] = MetricValue(
            mean=float(trigger) if trigger is not None else None,
            n=1,
            provenance=Provenance.EXISTING,
            note="Rounds until the retrain trigger fired",
        )
    else:
        kpis["drift_delay_rounds"] = MetricValue(note="No drift analysis for this scope")

    # --- Chart series ----------------------------------------------
    distributed = run.get("history_metrics_distributed", {}) or {}
    series: list[ChartSeries] = [
        _series_from_history(distributed.get("macro_f1"), "macro_f1", "Macro-F1 per round"),
        _series_from_history(distributed.get("accuracy"), "accuracy", "Accuracy per round"),
        _series_from_history(run.get("history_losses_centralized"), "val_mse", "Centralized val MSE", unit="MSE"),
    ]

    # Privacy-utility series (epsilon vs macro-F1) from the DP sweep.
    eps_points = []
    for eps in EPSILON_VALUES:
        dp_run = repository.load_dp_sweep(resolved, eps)
        if dp_run is None:
            continue
        dp_summary = summarize_per_client(dp_run["per_client_test_metrics"], rare_labels)
        value = dp_summary["macro_f1"]["mean"]
        # x is the epsilon index (inf first) -- the UI relabels the axis.
        eps_points.append(SeriesPoint(round=EPSILON_VALUES.index(eps), value=value))
    if eps_points:
        series.append(
            ChartSeries(
                key="privacy_utility",
                label="Macro-F1 vs epsilon (inf, 8, 3, 1, 0.5)",
                points=eps_points,
                provenance=Provenance.EXISTING,
            )
        )

    # Per-round MB/round is constant by construction (fixed parameter shapes).
    if mb_per_round is not None and distributed.get("macro_f1"):
        rounds = [int(e[0]) for e in distributed["macro_f1"]]
        series.append(
            ChartSeries(
                key="mb_per_round",
                label="Communication MB per round",
                points=[SeriesPoint(round=r, value=mb_per_round) for r in rounds],
                provenance=Provenance.DERIVED,
                unit="MB",
            )
        )

    # --- Recent events ---------------------------------------------
    events: list[EventItem] = []
    if total_rounds:
        events.append(
            EventItem(
                kind="round",
                title=f"Round {total_rounds} completed",
                detail=f"{run_name} finished its full schedule",
                round=int(total_rounds),
            )
        )
    if run.get("best_round") is not None:
        events.append(
            EventItem(
                kind="info",
                title=f"Best checkpoint at round {run['best_round']}",
                detail="Selected by lowest centralized validation MSE",
                round=int(run["best_round"]),
            )
        )
    if drift is not None and drift.get("first_trigger_round") is not None:
        events.append(
            EventItem(
                kind="drift",
                title="Drift detected",
                detail=f"Retrain trigger fired at round {drift['first_trigger_round']} "
                       f"({drift.get('num_adwin_drift_points', 0)} ADWIN change points)",
                round=int(drift["first_trigger_round"]),
            )
        )
    retrain = repository.load_drift_retrain(resolved)
    if retrain is not None:
        rounds_run = retrain.get("run_config", {}).get("num_retrain_rounds_run")
        events.append(
            EventItem(
                kind="retrain",
                title="Retraining completed",
                detail=f"{rounds_run} retrain rounds after the trigger",
            )
        )
    if repository.load_mechanism(resolved, "secagg") is not None:
        events.append(
            EventItem(kind="privacy", title="Secure aggregation run completed", detail="SecAgg+ (Flower)")
        )
    if achieved is not None:
        events.append(
            EventItem(
                kind="privacy",
                title="DP budget recorded",
                detail=f"Achieved epsilon {achieved:.4f} on the tightest completed run",
            )
        )

    model_status = {
        "run_name": run_name,
        "checkpoint_available": repository.checkpoint_exists(run_name, "best"),
        "checkpoint_path": f"{run_name}_best.pt",
        "num_classes": len(run_config.get("label_to_index", {})),
        "num_features": run_config.get("num_features"),
        "training_mode": "offline batch (not a live trainer)",
    }

    return DashboardResponse(
        scope=scope_info(dataset, scope),
        source_run=run_name,
        kpis=kpis,
        series=series,
        events=events,
        model_status=model_status,
    )
