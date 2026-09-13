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


def compute_zero_day_metrics(zero_day_predictions: np.ndarray, known_class_predictions: np.ndarray) -> dict:
    """Phase 7 (E4): does the NEW-CLASS mechanism actually work?

    `zero_day_predictions`: predictions (from classify_batch_with_prototypes,
    -1 = NEW CLASS) made on TRUE zero-day examples -- detection rate is
    how often these correctly come back -1.
    `known_class_predictions`: predictions made on TRUE known-class
    examples (val/test, never zero-day) -- false positive rate is how
    often these are WRONGLY flagged -1 (a known class mistaken for new).
    """
    n_zero_day = len(zero_day_predictions)
    n_known = len(known_class_predictions)

    detected = int((zero_day_predictions == -1).sum()) if n_zero_day else 0
    false_positives = int((known_class_predictions == -1).sum()) if n_known else 0

    detection_rate = detected / n_zero_day if n_zero_day else float("nan")
    false_positive_rate = false_positives / n_known if n_known else float("nan")

    # precision/recall/F1 framed as a binary "is this novel?" task
    true_positives = detected
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else float("nan")
    recall = detection_rate
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision == precision and recall == recall and (precision + recall) > 0)  # NaN-safe
        else float("nan")
    )

    return {
        "num_zero_day_samples": n_zero_day,
        "num_known_samples": n_known,
        "zero_day_detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_benign_false_positive_rate(confusion_matrix: list[list[int]], confusion_matrix_labels: list[str], benign_label: str = "BENIGN") -> float:
    """E1/T7's "FPR" column -- an IDS-standard false-positive rate, NOT
    a raw sklearn multiclass metric: of every TRUE BENIGN sequence, what
    fraction gets predicted as ANY non-BENIGN (attack) class, i.e. a
    false alarm. Derived directly from an already-computed confusion
    matrix (compute_classification_metrics's own output) -- no new
    forward pass or saved run needed.

    Returns NaN if this scope's classes don't include BENIGN at all
    (never fabricated as 0)."""
    if benign_label not in confusion_matrix_labels:
        return float("nan")
    cm = np.asarray(confusion_matrix)
    benign_idx = confusion_matrix_labels.index(benign_label)
    row = cm[benign_idx]
    total_true_benign = int(row.sum())
    if total_true_benign == 0:
        return float("nan")
    false_positives = total_true_benign - int(row[benign_idx])
    return false_positives / total_true_benign


def compute_unknown_vs_benign_detection_rate(zero_day_predictions: np.ndarray, benign_label_index: int) -> dict:
    """E4(a): "unknown-vs-benign detection rate" -- of the TRUE zero-day
    (novel-attack) examples, how many get predicted as ANYTHING OTHER
    THAN BENIGN, whether that's the correct NEW CLASS sentinel (-1, see
    compute_zero_day_metrics -- E4(b)'s stricter criterion) or an
    (incorrect) known-attack label. This is the weaker, "did the system
    at least notice this isn't normal traffic" criterion the frozen
    spec lists separately from (b)'s stricter "flagged as genuinely
    NEW, not just any anomaly" criterion -- distinct on purpose, not a
    duplicate: a model could route every zero-day example into some
    known ATTACK bucket (scoring well here, at 0.0 on (b)) or could
    genuinely recognize novelty (scoring well on both)."""
    n = len(zero_day_predictions)
    if n == 0:
        return {"num_zero_day_samples": 0, "unknown_vs_benign_detection_rate": float("nan")}
    not_benign = int((zero_day_predictions != benign_label_index).sum())
    return {
        "num_zero_day_samples": n,
        "unknown_vs_benign_detection_rate": not_benign / n,
    }
