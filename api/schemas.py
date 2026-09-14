"""Response models.

The `Provenance` tag on every metric is a research-integrity device, not
decoration: the UI renders it beside each number so a reader can always
tell a stored experiment result from a live computation from a demo
fallback. It mirrors the provenance key already published on the
project's results artifact.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Provenance(str, Enum):
    EXISTING = "existing"   # completed Phase 1-13 experiment, reused as-is
    NEW = "new"             # experiment run during the reporting pass
    DERIVED = "derived"     # computed from a saved confusion matrix/checkpoint
    LIVE = "live"           # computed just now, in-process (e.g. a prediction)
    DEMO = "demo"           # bundled fallback shown when the backend is offline


class MetricValue(BaseModel):
    """A metric with its uncertainty and sample size.

    `mean=None` means NOT MEASURED for this scope -- the UI renders an
    explicit empty state. It never means zero.
    """

    mean: float | None = None
    std: float | None = None
    n: int = 0
    provenance: Provenance = Provenance.EXISTING
    note: str | None = None


class ScopeInfo(BaseModel):
    dataset: str
    scope: str
    label: str
    num_clients: int
    clients_per_round: int
    num_features: int


class DatasetInfo(BaseModel):
    id: str
    label: str
    description: str
    scopes: list[str]
    scope_label: str
    rare_labels: list[str]
    zero_day_holdout: list[str]
    available: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    results_available: int
    checkpoints_available: int
    consolidated_tables: bool
    datasets: list[str]
    grafana_url: str
    prometheus_url: str
    notes: list[str] = Field(default_factory=list)


class SeriesPoint(BaseModel):
    round: int
    value: float | None


class ChartSeries(BaseModel):
    key: str
    label: str
    points: list[SeriesPoint]
    provenance: Provenance = Provenance.EXISTING
    unit: str | None = None


class EventItem(BaseModel):
    kind: Literal["info", "drift", "retrain", "privacy", "round", "warning"]
    title: str
    detail: str | None = None
    round: int | None = None
    provenance: Provenance = Provenance.EXISTING


class DashboardResponse(BaseModel):
    scope: ScopeInfo
    source_run: str
    kpis: dict[str, MetricValue]
    series: list[ChartSeries]
    events: list[EventItem]
    model_status: dict[str, Any]


class ExperimentSummary(BaseModel):
    id: str
    title: str
    subtitle: str
    table_ref: str
    available: bool


class ExperimentDetail(BaseModel):
    id: str
    title: str
    subtitle: str
    description: str
    interpretation: list[str]
    data: dict[str, Any]
    scopes_covered: list[str]
    provenance_note: str


class FLRunSummary(BaseModel):
    run_name: str
    mechanism: str
    label: str
    rounds: int
    best_round: int | None
    clients_configured: int
    clients_per_round: int
    local_epochs: int | None
    accuracy: MetricValue
    macro_f1: MetricValue
    mb_per_round: float | None


class FLRunDetail(BaseModel):
    run_name: str
    mechanism: str
    scope: ScopeInfo
    config: dict[str, Any]
    series: list[ChartSeries]
    clients: list[dict[str, Any]]
    best_round: int | None
    convergence_note: str


class PrivacyResponse(BaseModel):
    scope: ScopeInfo
    differential_privacy: dict[str, Any]
    secure_aggregation: dict[str, Any]
    utility_curve: list[dict[str, Any]]
    attacks: dict[str, Any]


class DriftStatusResponse(BaseModel):
    scope: ScopeInfo
    state: Literal["normal", "warning", "drift_detected", "retraining", "recovered", "unknown"]
    detector: dict[str, Any]
    recovery: dict[str, Any]
    rounds: list[dict[str, Any]]
    active_job: dict[str, Any] | None = None


class RetrainRequest(BaseModel):
    dataset: str
    scope: str | None = None
    num_retrain_rounds: int = Field(default=5, ge=1, le=20)


class RetrainJob(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "failed"]
    dataset: str
    scope: str
    run_name: str
    sandboxed: bool = True
    started_at: str
    finished_at: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None


class SampleItem(BaseModel):
    sequence_index: int
    label: str
    client_id: int
    split: str


class SamplesResponse(BaseModel):
    scope: ScopeInfo
    total: int
    samples: list[SampleItem]
    labels_available: list[str]


class PredictRequest(BaseModel):
    dataset: str
    scope: str | None = None
    sequence_index: int | None = None
    client_id: int | None = None
    mechanism: str = "personalized"
    window: list[list[float]] | None = None


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    scope: ScopeInfo
    model_run: str
    mechanism: str
    client_id: int | None
    source: Literal["prepared_sample", "uploaded_csv", "raw_window"]
    sequence_index: int | None
    true_label: str | None
    predicted_label: str
    confidence: float
    threat_level: Literal["benign", "low", "medium", "high", "unknown"]
    is_attack: bool
    is_new_class: bool
    new_class_distance: float | None
    new_class_threshold: float | None
    prototype_label: str | None
    reconstruction_mse: float
    top_k: list[dict[str, Any]]
    provenance: Provenance = Provenance.LIVE


class ErrorResponse(BaseModel):
    error: str
    detail: str
    hint: str | None = None
