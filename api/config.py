"""Paths, dataset/scope vocabulary, and the run-name conventions the
rest of the API resolves against.

The dataset/scope pairs here are the project's FROZEN experimental
scopes -- CICIDS2017 at three Dirichlet alphas and N-BaIoT's 45-client
scheme. They are deliberately modelled as two SEPARATE federations that
are never merged or aggregated together (a standing research-integrity
rule for this project), which is why every API response carries the
dataset+scope it belongs to rather than a single global "current" one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# The research modules live under src/ and are imported directly by the
# routers -- same sys.path convention every script/ entry point uses.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CONSOLIDATED_TABLES = RESULTS_DIR / "e1_e6_t7_tables.json"

# Prometheus' own exporter already owns 8000 (monitoring/metrics_exporter.py),
# so the API defaults to 8001 to avoid colliding with a live training run.
DEFAULT_API_PORT = 8001
GRAFANA_URL = "http://localhost:3000"
PROMETHEUS_URL = "http://localhost:9090"

# Vite dev server origins allowed through CORS.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@dataclass(frozen=True)
class Scope:
    """One frozen experimental scope: a dataset plus its partitioning."""

    dataset: str
    scope: str
    label: str
    num_clients: int
    clients_per_round: int
    num_features: int
    sequence_dir: Path
    client_id_col: str

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.scope}"

    @property
    def run_prefix(self) -> str:
        """How this scope's run files/checkpoints are named on disk."""
        return f"{self.dataset}_{self.scope}"


def _cicids_scope(alpha: str) -> Scope:
    return Scope(
        dataset="cicids2017",
        scope=alpha,
        # ASCII only: this string reaches the Windows console through
        # uvicorn's logger, which uses cp1252 and raises on a literal
        # alpha glyph. The frontend renders the proper symbol.
        label=f"CICIDS2017 alpha={alpha}",
        num_clients=40,
        clients_per_round=8,
        num_features=70,
        sequence_dir=PROJECT_ROOT / "data" / "processed" / "cicids2017" / "sequences" / f"alpha_{alpha}",
        client_id_col=f"client_id_alpha{alpha}",
    )


SCOPES: dict[str, Scope] = {
    "cicids2017:5": _cicids_scope("5"),
    "cicids2017:0.5": _cicids_scope("0.5"),
    "cicids2017:0.1": _cicids_scope("0.1"),
    "nbaiot:main_45": Scope(
        dataset="nbaiot",
        scope="main_45",
        label="N-BaIoT main_45",
        num_clients=45,
        clients_per_round=9,
        num_features=115,
        sequence_dir=PROJECT_ROOT / "data" / "processed" / "nbaiot" / "sequences" / "main_45",
        client_id_col="client_id_45",
    ),
}

DATASETS = {
    "cicids2017": {
        "id": "cicids2017",
        "label": "CICIDS2017",
        "description": "Network flow traffic, 40 clients partitioned by Dirichlet α, chronological Mon–Fri attack progression.",
        "scopes": ["5", "0.5", "0.1"],
        "scope_label": "Dirichlet α",
        "rare_labels": ["Heartbleed", "Infiltration", "Web Attack-SQL Injection", "Bot"],
        "zero_day_holdout": ["Infiltration", "Web Attack-Brute Force"],
    },
    "nbaiot": {
        "id": "nbaiot",
        "label": "N-BaIoT",
        "description": "IoT device telemetry, 9 devices × 5 shards = 45 clients, BASHLITE→Mirai attack progression.",
        "scopes": ["main_45"],
        "scope_label": "Partition scheme",
        "rare_labels": ["BASHLITE-Scan", "BASHLITE-Junk"],
        "zero_day_holdout": ["Mirai-Udpplain"],
    },
}

DEFAULT_SCOPE_KEY = "cicids2017:5"

# Mechanism -> the run-name suffix it is saved under. Used to resolve a
# scope + mechanism into an actual results/checkpoint file.
MECHANISM_SUFFIXES: dict[str, str] = {
    "centralized": "centralized",
    "fedavg": "fedavg",
    "fedprox": "fedprox_mu0.01_full",
    "base_paper_replication": "base_paper_replication_full",
    "personalized": "personalized",
    "secagg": "secagg_personalized",
    "dp_secagg": "dp_secagg_personalized_eps3.0_full",
}

EPSILON_VALUES = ["inf", "8", "3", "1", "0.5"]


def resolve_scope(dataset: str, scope: str | None = None) -> Scope:
    """Looks up a frozen scope, defaulting to the dataset's first scope.

    Raises KeyError with the valid options rather than silently falling
    back, so an invalid dataset/scope surfaces as a 404 instead of
    quietly rendering another scope's numbers.
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; valid: {sorted(DATASETS)}")
    if scope is None:
        scope = DATASETS[dataset]["scopes"][0]
    key = f"{dataset}:{scope}"
    if key not in SCOPES:
        raise KeyError(f"unknown scope {scope!r} for {dataset!r}; valid: {DATASETS[dataset]['scopes']}")
    return SCOPES[key]
