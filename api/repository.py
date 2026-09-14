"""Single read path for every stored experiment number the API serves.

Nothing here recomputes a metric that a completed run already reported.
Where a number is genuinely derived (rare-class recall and FPR from a
saved confusion matrix), it reuses the SAME evaluation helpers
scripts/build_e1_e6_tables.py uses, so the API and the thesis tables can
never disagree.

Missing data is returned as an explicit `None`/absent entry and surfaced
to the client as a "not measured" state -- never as a fabricated zero.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from api.config import (
    CHECKPOINT_DIR,
    CONSOLIDATED_TABLES,
    DATASETS,
    MECHANISM_SUFFIXES,
    RESULTS_DIR,
    Scope,
)

from fedpda_ids.evaluation.metrics import (
    compute_benign_false_positive_rate,
    extract_rare_class_metrics,
)


class ResultsRepository:
    """Loads and caches `experiments/results/*.json`.

    Files are cached by (path, mtime) so a newly-finished run -- e.g. a
    sandboxed retrain triggered from the UI -- is picked up without a
    server restart, while repeated reads of a stable file stay cheap.
    """

    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        self.results_dir = results_dir

    def path_for(self, run_name: str) -> Path:
        return self.results_dir / f"{run_name}.json"

    def exists(self, run_name: str) -> bool:
        return self.path_for(run_name).exists()

    def load(self, run_name: str) -> dict[str, Any] | None:
        path = self.path_for(run_name)
        if not path.exists():
            return None
        return _load_json_cached(str(path), path.stat().st_mtime_ns)

    def load_consolidated_tables(self) -> dict[str, Any] | None:
        """The E1–E6/T7 source of truth (scripts/build_e1_e6_tables.py)."""
        if not CONSOLIDATED_TABLES.exists():
            return None
        return _load_json_cached(str(CONSOLIDATED_TABLES), CONSOLIDATED_TABLES.stat().st_mtime_ns)

    def run_name(self, scope: Scope, mechanism: str) -> str:
        suffix = MECHANISM_SUFFIXES.get(mechanism, mechanism)
        return f"{scope.run_prefix}_{suffix}"

    def load_mechanism(self, scope: Scope, mechanism: str) -> dict[str, Any] | None:
        return self.load(self.run_name(scope, mechanism))

    def load_dp_sweep(self, scope: Scope, epsilon: str) -> dict[str, Any] | None:
        """One point of the DP epsilon sweep. `inf` maps to the non-DP
        personalized run, which IS the epsilon=infinity point (no separate
        run exists, by design -- see privacy/dp.py's module docstring)."""
        if epsilon in ("inf", "∞"):
            return self.load_mechanism(scope, "personalized")
        return self.load(f"{scope.run_prefix}_dp_personalized_eps{_eps_file_token(epsilon)}")

    def load_prototypes(self, scope: Scope, epsilon: str = "inf") -> dict[str, Any] | None:
        if epsilon in ("inf", "∞"):
            return self.load(f"{scope.run_prefix}_prototypes")
        return self.load(f"{scope.run_prefix}_prototypes_eps{epsilon}")

    def load_drift(self, scope: Scope) -> dict[str, Any] | None:
        return self.load(f"{scope.run_prefix}_drift")

    def load_drift_retrain(self, scope: Scope) -> dict[str, Any] | None:
        for candidate in (f"{scope.run_prefix}_drift_retrain_full", f"{scope.run_prefix}_drift_retrain"):
            found = self.load(candidate)
            if found is not None:
                return found
        return None

    def load_mia(self, run_name: str, suffix: str = "last") -> dict[str, Any] | None:
        return self.load(f"mia_{run_name}_{suffix}")

    def list_runs(self, scope: Scope) -> list[str]:
        """Every completed run file belonging to this scope, by run name."""
        prefix = f"{scope.run_prefix}_"
        names = []
        for path in sorted(self.results_dir.glob(f"{prefix}*.json")):
            names.append(path.stem)
        return names

    def checkpoint_exists(self, run_name: str, suffix: str = "best") -> bool:
        return (CHECKPOINT_DIR / f"{run_name}_{suffix}.pt").exists()


@lru_cache(maxsize=256)
def _load_json_cached(path: str, _mtime_ns: int) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _eps_file_token(epsilon: str) -> str:
    """DP sweep files are named eps8.0/eps3.0/eps1.0/eps0.5 -- normalise
    whatever the client sent ("8", "8.0", 8) onto that convention."""
    value = float(epsilon)
    return f"{value:.1f}"


# ---------------------------------------------------------------------
# Derived metrics -- computed from already-saved artifacts only.
# ---------------------------------------------------------------------


def safe_float(value: Any) -> float | None:
    """JSON cannot carry NaN/Infinity. Anything non-finite becomes None
    and renders as an explicit "not measured" state in the UI, rather
    than silently becoming 0 or breaking the response."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def mean_std(values: list[float]) -> dict[str, Any]:
    """Same aggregation convention scripts/build_e1_e6_tables.py uses:
    a single observation reports std=None (never a fabricated 0.0)."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    if len(clean) == 1:
        return {"mean": float(clean[0]), "std": None, "n": 1}
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    return {"mean": float(mean), "std": float(variance ** 0.5), "n": len(clean)}


def rare_recall(metrics: dict[str, Any], rare_labels: list[str]) -> float | None:
    """Mean recall across whichever rare labels this scope's classes
    actually include -- absent labels are excluded, never counted as 0."""
    rare = extract_rare_class_metrics(metrics, rare_labels)
    present = [v["recall"] for v in rare.values() if "status" not in v]
    if not present:
        return None
    return float(sum(present) / len(present))


def benign_fpr(metrics: dict[str, Any]) -> float | None:
    if "confusion_matrix" not in metrics or "confusion_matrix_labels" not in metrics:
        return None
    return safe_float(
        compute_benign_false_positive_rate(metrics["confusion_matrix"], metrics["confusion_matrix_labels"])
    )


def summarize_per_client(per_client: dict[str, Any], rare_labels: list[str]) -> dict[str, Any]:
    """Aggregates a run's per-client test metrics the way every table in
    this project already does: mean±std ACROSS clients."""
    valid = [
        m for m in per_client.values()
        if isinstance(m, dict) and "status" not in m and "confusion_matrix" in m
    ]
    return {
        "accuracy": mean_std([m["accuracy"] for m in valid]),
        "macro_f1": mean_std([m["macro_f1"] for m in valid]),
        "rare_class_recall": mean_std([r for r in (rare_recall(m, rare_labels) for m in valid) if r is not None]),
        "fpr": mean_std([f for f in (benign_fpr(m) for m in valid) if f is not None]),
        "num_clients": len(valid),
    }


def summarize_global(test_metrics: dict[str, Any], rare_labels: list[str]) -> dict[str, Any]:
    """Single-global-model runs (Centralized, FedAvg, FedProx): one value
    per metric, std=None because there is no cross-client distribution."""
    return {
        "accuracy": {"mean": safe_float(test_metrics.get("accuracy")), "std": None, "n": 1},
        "macro_f1": {"mean": safe_float(test_metrics.get("macro_f1")), "std": None, "n": 1},
        "rare_class_recall": {"mean": rare_recall(test_metrics, rare_labels), "std": None, "n": 1},
        "fpr": {"mean": benign_fpr(test_metrics), "std": None, "n": 1},
        "num_clients": 1,
    }


def rare_labels_for(dataset: str) -> list[str]:
    return list(DATASETS[dataset]["rare_labels"])


repository = ResultsRepository()
