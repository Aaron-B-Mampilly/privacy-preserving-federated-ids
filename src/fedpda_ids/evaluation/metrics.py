"""Phase 4: classification metrics.

Per the frozen spec: never report accuracy alone. Every call here
returns accuracy alongside precision/recall/macro-F1/weighted-F1/
per-class breakdown/confusion matrix in one dict, so it's structurally
awkward to log just accuracy by accident.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, index_to_label: dict[int, str]
) -> dict:
    """y_true/y_pred: integer class indices. index_to_label: the scope's
    class-index mapping (inverted), used to label the per-class and
    confusion-matrix output by name instead of a bare integer."""
    labels = sorted(index_to_label.keys())
    label_names = [index_to_label[i] for i in labels]

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        label_names[i]: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }
        for i in range(len(labels))
    }

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names,
        "num_samples": int(len(y_true)),
    }


def extract_rare_class_metrics(metrics: dict, rare_labels: list[str]) -> dict:
    """Pulls out just the rare-class rows from an already-computed
    metrics dict (see compute_classification_metrics) -- labels not
    present in this scope's classes are reported as absent, not
    silently skipped."""
    result = {}
    for label in rare_labels:
        if label in metrics["per_class"]:
            result[label] = metrics["per_class"][label]
        else:
            result[label] = {"status": "not_in_scope_classes"}
    return result
