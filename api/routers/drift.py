"""Drift monitoring and the (real, sandboxed) retrain trigger.

POST /drift/retrain genuinely runs `run_drift_triggered_retraining` --
the same function scripts/run_drift_triggered_retraining.py calls. It is
made safe for a UI button in exactly one way: the output run name is
always suffixed `_uidemo_<timestamp>`, so a UI-triggered retrain can
never overwrite an official experiment's checkpoint or results file. The
source checkpoint it starts from is opened read-only.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone

import pandas as pd
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from torch.utils.data import DataLoader

from api.config import CHECKPOINT_DIR, resolve_scope
from api.registry import registry
from api.repository import repository, safe_float
from api.routers.health import scope_info
from api.schemas import DriftStatusResponse, RetrainJob, RetrainRequest

from fedpda_ids.data.sequence_dataset import LabeledSequenceIndexDataset
from fedpda_ids.drift.retrain import split_client_rows_at_cutoff, trigger_cutoff_timestamp
from fedpda_ids.federated.client import load_local_head, local_head_path
from fedpda_ids.federated.simulation import (
    build_trainable_client_pool,
    load_shared_checkpoint,
    make_model,
    run_drift_triggered_retraining,
)
from fedpda_ids.models.trainer import evaluate

router = APIRouter(prefix="/drift", tags=["drift"])

# In-process job store. Retrains are short (under ~2 minutes) and this is
# a single-process demo API, so a dict + lock is the right weight here --
# no external queue or broker is warranted.
_jobs: dict[str, RetrainJob] = {}
_jobs_lock = threading.Lock()


@router.get("/status", response_model=DriftStatusResponse)
def status(
    dataset: str = Query(...),
    scope: str | None = Query(None),
) -> DriftStatusResponse:
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    drift = repository.load_drift(resolved)
    if drift is None:
        raise HTTPException(
            status_code=404,
            detail=f"No drift analysis for {resolved.label}. Run scripts/detect_drift.py first.",
        )

    cfg = drift.get("run_config", {})
    reports = drift.get("round_reports", []) or []
    trigger_round = drift.get("first_trigger_round")

    detector = {
        "primary": "ADWIN",
        "backup": "PSI (mean across latent dimensions)",
        "adwin_delta": cfg.get("adwin_delta"),
        "psi_threshold": cfg.get("psi_threshold"),
        "round_window_size": cfg.get("round_window_size"),
        "consecutive_rounds_for_retrain": cfg.get("consecutive_rounds_for_retrain"),
        "num_adwin_drift_points": drift.get("num_adwin_drift_points"),
        "num_rounds": drift.get("num_rounds"),
        "reference_reconstruction_mse": drift.get("reference_reconstruction_mse"),
        "stream_reconstruction_mse": drift.get("stream_reconstruction_mse"),
        "first_trigger_round": trigger_round,
        "provenance": "existing",
    }

    retrain = repository.load_drift_retrain(resolved)
    if retrain is not None:
        # The matched-subset comparison is computed by
        # scripts/build_e1_e6_tables.py (it is not written into the raw
        # retrain run file), so it is read from the consolidated E5 table.
        # It matters: without it the three arms can describe different
        # client subsets and retraining looks harmful when it is not.
        tables = repository.load_consolidated_tables() or {}
        e5_key = "cicids2017_alpha5" if resolved.key == "cicids2017:5" else (
            "nbaiot_main_45" if resolved.dataset == "nbaiot" else None
        )
        e5_entry = (tables.get("E5_drift_and_recovery", {}) or {}).get(e5_key, {}) if e5_key else {}
        matched = e5_entry.get("matched_subset_comparison", {}) if isinstance(e5_entry, dict) else {}
        recovery = {
            "available": True,
            "f1_before_drift": retrain.get("f1_before_drift"),
            "f1_after_drift_no_adaptation": retrain.get("f1_after_drift_no_adaptation_without_monitor"),
            "f1_after_retraining": retrain.get("f1_after_triggered_retraining_with_monitor"),
            "detection_delay_rounds": retrain.get("drift_detection_delay_rounds"),
            "num_retrain_rounds_run": retrain.get("run_config", {}).get("num_retrain_rounds_run"),
            "matched_subset": matched,
            "matched_subset_note": matched.get("note"),
            "provenance": "new",
        }
        state = "recovered"
    else:
        recovery = {
            "available": False,
            "status": "pending",
            "reason": "No completed retrain run for this scope yet -- trigger one to generate it.",
        }
        state = "drift_detected" if trigger_round is not None else "normal"

    # Trim the per-round detector trace; the UI charts a window, not 148 rows.
    rounds = [
        {
            "round": r.get("round"),
            "psi": safe_float(r.get("psi")),
            "psi_flag": r.get("psi_flag"),
            "adwin_flag": r.get("adwin_flag"),
            "mean_drift": safe_float(r.get("mean_drift")),
            "retrain_triggered": r.get("retrain_triggered"),
        }
        for r in reports[:120]
    ]

    with _jobs_lock:
        active = next(
            (j for j in _jobs.values()
             if j.dataset == resolved.dataset and j.scope == resolved.scope
             and j.state in ("queued", "running")),
            None,
        )
        if active is not None:
            state = "retraining"

    return DriftStatusResponse(
        scope=scope_info(dataset, scope),
        state=state,
        detector=detector,
        recovery=recovery,
        rounds=rounds,
        active_job=active.model_dump() if active else None,
    )


@router.get("/events")
def events(
    dataset: str = Query(...),
    scope: str | None = Query(None),
) -> dict:
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    drift = repository.load_drift(resolved)
    if drift is None:
        return {"scope": scope_info(dataset, scope).model_dump(), "events": []}

    timeline = [{
        "stage": "normal",
        "label": "Monitoring",
        "detail": f"Reference: last {drift.get('run_config', {}).get('num_reference_sequences', '?')} "
                  "benign training windows",
        "round": 0,
    }]
    trigger = drift.get("first_trigger_round")
    if trigger is not None:
        timeline.append({
            "stage": "drift_detected",
            "label": "Drift detected",
            "detail": f"Trigger fired after {trigger} round(s) of consecutive drift",
            "round": trigger,
        })
    retrain = repository.load_drift_retrain(resolved)
    if retrain is not None:
        rounds_run = retrain.get("run_config", {}).get("num_retrain_rounds_run")
        timeline.append({
            "stage": "retraining",
            "label": "Retraining triggered",
            "detail": f"{rounds_run} rounds on data collected before the cutoff",
            "round": trigger,
        })
        matched = retrain.get("matched_subset_comparison", {})
        before = (matched.get("f1_before_drift") or {}).get("mean")
        after_no = (matched.get("f1_after_drift_no_adaptation") or {}).get("mean")
        after_re = (matched.get("f1_after_triggered_retraining") or {}).get("mean")
        timeline.append({
            "stage": "recovered",
            "label": "Recovery measured",
            "detail": (
                f"F1 {before:.3f} before drift, {after_no:.3f} without adaptation, "
                f"{after_re:.3f} after retraining"
                if None not in (before, after_no, after_re) else "Recovery metrics recorded"
            ),
            "round": None,
        })
    return {"scope": scope_info(dataset, scope).model_dump(), "events": timeline}


@router.post("/retrain", response_model=RetrainJob)
def trigger_retrain(request: RetrainRequest, background: BackgroundTasks) -> RetrainJob:
    try:
        resolved = resolve_scope(request.dataset, request.scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    drift = repository.load_drift(resolved)
    if drift is None or drift.get("first_trigger_round") is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No drift trigger recorded for {resolved.label}, so there is nothing to retrain "
                "from. Run scripts/detect_drift.py for this scope first."
            ),
        )

    source_run = repository.run_name(resolved, "personalized")
    if not repository.checkpoint_exists(source_run, "best"):
        raise HTTPException(
            status_code=409,
            detail=f"Source checkpoint {source_run}_best.pt is missing; cannot retrain from it.",
        )

    with _jobs_lock:
        if any(j.dataset == resolved.dataset and j.scope == resolved.scope
               and j.state in ("queued", "running") for j in _jobs.values()):
            raise HTTPException(status_code=409, detail="A retrain is already running for this scope.")

    job_id = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    # SANDBOX: the _uidemo_ suffix is what makes this safe to expose --
    # an official run name can never be produced by this endpoint.
    sandbox_run = f"{source_run}_uidemo_{stamp}"

    job = RetrainJob(
        job_id=job_id,
        state="queued",
        dataset=resolved.dataset,
        scope=resolved.scope,
        run_name=sandbox_run,
        started_at=datetime.now(timezone.utc).isoformat(),
        message="Queued",
    )
    with _jobs_lock:
        _jobs[job_id] = job

    background.add_task(_run_retrain, job_id, resolved, source_run, sandbox_run, request.num_retrain_rounds)
    return job


@router.get("/retrain/{job_id}", response_model=RetrainJob)
def retrain_status(job_id: str) -> RetrainJob:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No retrain job {job_id!r}")
    return job


def _update(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        _jobs[job_id] = job.model_copy(update=fields)


def _run_retrain(job_id: str, resolved, source_run: str, sandbox_run: str, num_rounds: int) -> None:
    """Executes the real retrain pipeline in a background thread."""
    _update(job_id, state="running", message="Preparing client data splits")
    try:
        cfg = registry.config
        drift = repository.load_drift(resolved)
        source_results = repository.load_mechanism(resolved, "personalized")
        label_to_index = source_results["run_config"]["label_to_index"]
        index_to_label = {int(i): label for label, i in label_to_index.items()}
        num_classes = len(label_to_index)
        batch_size = cfg["model"]["batch_size"]
        device = torch.device("cpu")

        pool = build_trainable_client_pool(
            resolved.sequence_dir, resolved.client_id_col, resolved.num_clients,
            cfg["training"]["local_training"]["min_train_sequences"],
            cfg["training"]["local_training"]["min_train_classes"],
        )

        metadata = pd.read_parquet(
            resolved.sequence_dir / "metadata.parquet",
            columns=["temporal_split", "sequence_label", "sequence_index",
                     "window_start_time", resolved.client_id_col],
        )
        pooled_test = metadata[
            (metadata["temporal_split"] == "test") & (metadata[resolved.client_id_col] != -1)
        ].sort_values("window_start_time")
        cutoff = trigger_cutoff_timestamp(
            pooled_test["window_start_time"].to_numpy(),
            round_window_size=drift["run_config"]["round_window_size"],
            first_trigger_round=drift["first_trigger_round"],
        )

        retrain_loaders, eval_loaders = {}, {}
        for client_id in pool:
            rows = metadata[
                (metadata["temporal_split"] == "test") & (metadata[resolved.client_id_col] == client_id)
            ]
            before, after = split_client_rows_at_cutoff(rows, "window_start_time", cutoff)
            retrain_ds = LabeledSequenceIndexDataset(
                resolved.sequence_dir, before["sequence_index"].to_numpy(),
                before["sequence_label"].to_numpy(), label_to_index,
            )
            eval_ds = LabeledSequenceIndexDataset(
                resolved.sequence_dir, after["sequence_index"].to_numpy(),
                after["sequence_label"].to_numpy(), label_to_index,
            )
            retrain_loaders[client_id] = DataLoader(
                retrain_ds, batch_size=batch_size, shuffle=len(retrain_ds) > 0
            )
            eval_loaders[client_id] = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

        _update(job_id, message=f"Retraining {num_rounds} rounds over {len(pool)} clients")
        result = run_drift_triggered_retraining(
            pool=pool,
            retrain_loaders=retrain_loaders,
            checkpoint_dir=CHECKPOINT_DIR,
            source_run_name=source_run,
            new_run_name=sandbox_run,
            num_features=resolved.num_features,
            num_classes=num_classes,
            model_cfg=cfg["model"],
            window_size=cfg["sequence"]["window_size"],
            local_epochs=cfg["federated"]["local_epochs"],
            num_retrain_rounds=num_rounds,
            lambda_ce=cfg["model"]["loss"]["lambda_ce"],
            learning_rate=cfg["model"]["optimizer"]["learning_rate"],
            device=device,
        )

        _update(job_id, message="Evaluating recovery on the held-out post-cutoff slice")
        before_scores, after_scores = [], []
        for client_id in result["clients_retrained"]:
            loader = eval_loaders.get(client_id)
            if loader is None or len(loader.dataset) == 0:
                continue
            # Before: original checkpoint + that client's original head.
            base = make_model(resolved.num_features, num_classes, cfg["model"], cfg["sequence"]["window_size"])
            load_shared_checkpoint(CHECKPOINT_DIR / f"{source_run}_best.pt", base)
            if not load_local_head(base, local_head_path(CHECKPOINT_DIR, source_run, client_id)):
                continue
            before_scores.append(
                evaluate(base, loader, device, cfg["model"]["loss"]["lambda_ce"], index_to_label)["macro_f1"]
            )
            # After: retrained checkpoint + that client's retrained head.
            tuned = make_model(resolved.num_features, num_classes, cfg["model"], cfg["sequence"]["window_size"])
            load_shared_checkpoint(CHECKPOINT_DIR / f"{sandbox_run}_best.pt", tuned)
            if not load_local_head(tuned, local_head_path(CHECKPOINT_DIR, sandbox_run, client_id)):
                continue
            after_scores.append(
                evaluate(tuned, loader, device, cfg["model"]["loss"]["lambda_ce"], index_to_label)["macro_f1"]
            )

        def _mean(values):
            return float(sum(values) / len(values)) if values else None

        _update(
            job_id,
            state="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            message="Retraining complete",
            result={
                "run_name": sandbox_run,
                "sandboxed": True,
                "num_retrain_rounds_run": result["num_retrain_rounds_run"],
                "clients_retrained": result["clients_retrained"],
                "num_clients_retrained": len(result["clients_retrained"]),
                "matched_clients_evaluated": len(after_scores),
                "macro_f1_before_retrain": _mean(before_scores),
                "macro_f1_after_retrain": _mean(after_scores),
                "evaluation_note": (
                    "Both numbers are computed on the SAME held-out post-cutoff slice for the SAME "
                    "clients, so the comparison is matched -- the pitfall that made retraining look "
                    "harmful in the first E5 analysis."
                ),
                "provenance": "live",
            },
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a job failure
        _update(
            job_id,
            state="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            message=f"{type(exc).__name__}: {exc}",
            result={"traceback_summary": traceback.format_exc(limit=3).splitlines()[-1]},
        )
