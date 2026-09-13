"""Phase 13: Prometheus metrics export for live FL training observability.

Exposes round-level training/evaluation metrics via an HTTP endpoint
(prometheus_client's start_http_server) that Prometheus scrapes on an
interval, visualized in Grafana. Purely additive instrumentation --
recording a metric never affects training/aggregation logic, and the
HTTP server is opt-in (start_metrics_server must be called explicitly)
so importing this module, or running any test, never binds a port.

Scope (deliberately, not from the frozen spec's own detail -- it names
the phase but not which metrics): round number, centralized validation
loss/MSE (the primary "is training actually progressing" signal shared
by every FL mechanism in this project), and the DP mechanism's
per-round adaptive clip norm (unique to Phase 8's runs). Per-client
distributed metrics are already fully captured in each run's own
results JSON at the end of a run; live-monitoring's practical value is
mainly "is this multi-hour run stuck or progressing," which centralized
val loss + round number already answers well without instrumenting
every strategy variant's aggregate_evaluate path too.
"""

from __future__ import annotations

from prometheus_client import Gauge, start_http_server

FL_ROUND = Gauge("fedpda_fl_round", "Current FL round number", ["run_name"])
FL_CENTRALIZED_VAL_LOSS = Gauge(
    "fedpda_centralized_val_loss", "Centralized validation loss/MSE this round", ["run_name"]
)
DP_CLIP_NORM = Gauge(
    "fedpda_dp_clip_norm", "Adaptive median clip norm this round (DP runs only)", ["run_name"]
)

_server_started = False


def start_metrics_server(port: int = 8000) -> None:
    """Starts the Prometheus HTTP server exactly once per process --
    idempotent (safe to call from every training script's main()) and
    never called implicitly by anything in this module."""
    global _server_started
    if not _server_started:
        start_http_server(port)
        _server_started = True


def record_round(run_name: str, round_num: int, val_loss: float | None = None) -> None:
    FL_ROUND.labels(run_name=run_name).set(round_num)
    if val_loss is not None:
        FL_CENTRALIZED_VAL_LOSS.labels(run_name=run_name).set(val_loss)


def record_dp_clip_norm(run_name: str, clip_norm: float) -> None:
    DP_CLIP_NORM.labels(run_name=run_name).set(clip_norm)
