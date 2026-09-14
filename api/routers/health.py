"""Health + dataset vocabulary. The frontend polls /health to drive its
connected / connecting / disconnected indicator."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.config import (
    CHECKPOINT_DIR,
    DATASETS,
    GRAFANA_URL,
    PROMETHEUS_URL,
    RESULTS_DIR,
    SCOPES,
    resolve_scope,
)
from api.repository import repository
from api.schemas import DatasetInfo, HealthResponse, ScopeInfo

router = APIRouter(tags=["system"])

API_VERSION = "1.0.0"


def scope_info(dataset: str, scope: str | None) -> ScopeInfo:
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScopeInfo(
        dataset=resolved.dataset,
        scope=resolved.scope,
        label=resolved.label,
        num_clients=resolved.num_clients,
        clients_per_round=resolved.clients_per_round,
        num_features=resolved.num_features,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    results = list(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    checkpoints = list(CHECKPOINT_DIR.glob("*.pt")) if CHECKPOINT_DIR.exists() else []
    tables = repository.load_consolidated_tables() is not None

    notes: list[str] = []
    if not tables:
        notes.append("Consolidated E1-E6/T7 tables not found -- run scripts/build_e1_e6_tables.py.")
    if not checkpoints:
        notes.append("No checkpoints found -- prediction endpoints will be unavailable.")

    missing_sequences = [s.label for s in SCOPES.values() if not (s.sequence_dir / "X.npy").exists()]
    if missing_sequences:
        notes.append(f"Sequence artifacts missing for: {', '.join(missing_sequences)}")

    return HealthResponse(
        status="ok" if (tables and checkpoints and not missing_sequences) else "degraded",
        version=API_VERSION,
        results_available=len(results),
        checkpoints_available=len(checkpoints),
        consolidated_tables=tables,
        datasets=sorted(DATASETS),
        grafana_url=GRAFANA_URL,
        prometheus_url=PROMETHEUS_URL,
        notes=notes,
    )


@router.get("/datasets", response_model=list[DatasetInfo])
def datasets() -> list[DatasetInfo]:
    """The two federations, always separate. There is deliberately no
    'all datasets' option -- merging them is a standing project rule."""
    out = []
    for meta in DATASETS.values():
        available = any(
            (SCOPES[f"{meta['id']}:{s}"].sequence_dir / "X.npy").exists()
            for s in meta["scopes"]
            if f"{meta['id']}:{s}" in SCOPES
        )
        out.append(DatasetInfo(**meta, available=available))
    return out
