"""E1/T7's FPR column: compute_benign_false_positive_rate, derived
directly from an already-computed confusion matrix -- no new forward
pass needed. See metrics.py's docstring for why this is an IDS-standard
"false alarm rate on true-benign traffic" definition, not a raw
sklearn multiclass metric.
"""

import numpy as np

from fedpda_ids.evaluation.metrics import compute_benign_false_positive_rate, compute_classification_metrics


def test_benign_fpr_from_direct_confusion_matrix():
    # labels: BENIGN, ATTACK -- 10 true BENIGN, 2 misclassified as ATTACK
    cm = [[8, 2], [1, 9]]
    labels = ["BENIGN", "ATTACK"]
    fpr = compute_benign_false_positive_rate(cm, labels)
    assert np.isclose(fpr, 2 / 10)


def test_benign_fpr_zero_when_no_false_alarms():
    cm = [[10, 0], [3, 7]]
    labels = ["BENIGN", "ATTACK"]
    assert compute_benign_false_positive_rate(cm, labels) == 0.0


def test_benign_fpr_missing_benign_label_is_nan():
    cm = [[5, 1], [0, 9]]
    labels = ["ATTACK_A", "ATTACK_B"]
    assert np.isnan(compute_benign_false_positive_rate(cm, labels))


def test_benign_fpr_no_true_benign_samples_is_nan():
    cm = [[0, 0], [4, 6]]
    labels = ["BENIGN", "ATTACK"]
    assert np.isnan(compute_benign_false_positive_rate(cm, labels))


def test_benign_fpr_matches_compute_classification_metrics_output():
    # end-to-end: feed real y_true/y_pred through compute_classification_metrics,
    # then compute FPR from its confusion_matrix/confusion_matrix_labels output.
    index_to_label = {0: "BENIGN", 1: "ATTACK"}
    y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0])  # 2/4 true-benign misclassified
    metrics = compute_classification_metrics(y_true, y_pred, index_to_label)
    fpr = compute_benign_false_positive_rate(metrics["confusion_matrix"], metrics["confusion_matrix_labels"])
    assert np.isclose(fpr, 2 / 4)
