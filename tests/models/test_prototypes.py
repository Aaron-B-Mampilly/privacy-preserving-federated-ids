"""Phase 7: latent class prototype mechanism tests."""

import numpy as np
import torch

from fedpda_ids.evaluation.metrics import compute_zero_day_metrics
from fedpda_ids.models.prototypes import (
    aggregate_prototypes,
    calibrate_threshold,
    classify_batch_with_prototypes,
    classify_with_prototypes,
    client_prototypes,
    clip_prototype,
    compute_class_prototypes,
    extract_latents,
)


# ---------------------------------------------------------------------
# extract_latents / compute_class_prototypes
# ---------------------------------------------------------------------


class _IdentityEncoder(torch.nn.Module):
    """Returns the last timestep of the input, unchanged -- lets tests
    control exact latent values instead of depending on a trained LSTM."""

    def forward(self, x):
        return x[:, -1, :]


def _make_loader(latents_by_label: dict[int, list[list[float]]], window=10, num_features=2):
    xs, ys = [], []
    for label, vectors in latents_by_label.items():
        for v in vectors:
            # build a (window, F) sequence whose LAST row is exactly v
            seq = np.tile(np.array(v, dtype=np.float32), (window, 1))
            xs.append(seq)
            ys.append(label)
    if xs:
        x = torch.tensor(np.stack(xs), dtype=torch.float32)
        y = torch.tensor(ys, dtype=torch.long)
    else:
        x = torch.empty((0, window, num_features), dtype=torch.float32)
        y = torch.empty((0,), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def test_extract_latents_and_compute_class_prototypes_are_exact_means():
    loader = _make_loader({0: [[0.0, 0.0], [2.0, 0.0]], 1: [[10.0, 10.0], [10.0, 12.0]]})
    encoder = _IdentityEncoder()
    z, labels = extract_latents(encoder, loader, torch.device("cpu"))
    protos = compute_class_prototypes(z, labels)

    assert np.allclose(protos[0][0], [1.0, 0.0])
    assert protos[0][1] == 2
    assert np.allclose(protos[1][0], [10.0, 11.0])
    assert protos[1][1] == 2


# ---------------------------------------------------------------------
# clip_prototype
# ---------------------------------------------------------------------


def test_clip_prototype_leaves_small_vectors_unchanged():
    v = np.array([0.1, 0.2])
    assert np.allclose(clip_prototype(v, bound=1.0), v)


def test_clip_prototype_shrinks_large_vectors_to_bound_norm():
    v = np.array([3.0, 4.0])  # norm = 5
    clipped = clip_prototype(v, bound=1.0)
    assert np.isclose(np.linalg.norm(clipped), 1.0)
    assert np.allclose(clipped, v / 5.0)  # direction preserved


def test_clip_prototype_zero_vector_stays_zero():
    v = np.zeros(3)
    assert np.allclose(clip_prototype(v, bound=1.0), v)


# ---------------------------------------------------------------------
# client_prototypes (noise_fn plumbing -- Phase 8 placeholder)
# ---------------------------------------------------------------------


def test_client_prototypes_applies_clipping_and_optional_noise():
    loader = _make_loader({0: [[3.0, 4.0]]})  # norm 5, will get clipped
    encoder = _IdentityEncoder()

    protos_no_noise = client_prototypes(encoder, loader, torch.device("cpu"), clip_bound=1.0)
    assert np.isclose(np.linalg.norm(protos_no_noise[0][0]), 1.0)

    def add_one(v):
        return v + 1.0

    protos_with_noise = client_prototypes(encoder, loader, torch.device("cpu"), clip_bound=1.0, noise_fn=add_one)
    assert np.allclose(protos_with_noise[0][0], protos_no_noise[0][0] + 1.0)


def test_client_prototypes_empty_loader_returns_empty_dict():
    loader = _make_loader({})
    encoder = _IdentityEncoder()
    protos = client_prototypes(encoder, loader, torch.device("cpu"))
    assert protos == {}


# ---------------------------------------------------------------------
# aggregate_prototypes
# ---------------------------------------------------------------------


def test_aggregate_prototypes_is_support_weighted_mean():
    client_a = {0: (np.array([0.0, 0.0]), 1)}
    client_b = {0: (np.array([4.0, 0.0]), 3)}  # 3x the weight of client_a
    aggregated = aggregate_prototypes([client_a, client_b])
    # weighted mean: (0*1 + 4*3) / (1+3) = 3.0
    assert np.allclose(aggregated[0], [3.0, 0.0])


def test_aggregate_prototypes_handles_classes_present_in_only_some_clients():
    client_a = {0: (np.array([1.0]), 2)}
    client_b = {1: (np.array([5.0]), 2)}
    aggregated = aggregate_prototypes([client_a, client_b])
    assert set(aggregated.keys()) == {0, 1}
    assert np.allclose(aggregated[0], [1.0])
    assert np.allclose(aggregated[1], [5.0])


# ---------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------


def test_calibrate_threshold_is_half_the_weighted_mean_pairwise_distance():
    prototypes = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
    weights = {0: 1, 1: 1}
    tau = calibrate_threshold(prototypes, weights, multiplier=0.5)
    assert np.isclose(tau, 5.0)  # distance=10, weighted mean=10, *0.5 = 5


def test_calibrate_threshold_requires_at_least_two_classes():
    import pytest

    with pytest.raises(ValueError):
        calibrate_threshold({0: np.array([0.0])}, {0: 1})


# ---------------------------------------------------------------------
# classify_with_prototypes / classify_batch_with_prototypes
# ---------------------------------------------------------------------


def test_classify_with_prototypes_known_class_returns_nearest_label():
    prototypes = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
    label, dist = classify_with_prototypes(np.array([0.5, 0.0]), prototypes, threshold=5.0)
    assert label == 0
    assert np.isclose(dist, 0.5)


def test_classify_with_prototypes_far_point_is_new_class():
    prototypes = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
    label, dist = classify_with_prototypes(np.array([100.0, 100.0]), prototypes, threshold=5.0)
    assert label is None  # NEW CLASS


def test_classify_batch_matches_single_example_classify():
    prototypes = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
    threshold = 5.0
    batch = np.array([[0.5, 0.0], [100.0, 100.0], [9.5, 0.0]])
    preds, dists = classify_batch_with_prototypes(batch, prototypes, threshold)
    assert preds.tolist() == [0, -1, 1]

    for i in range(len(batch)):
        single_label, single_dist = classify_with_prototypes(batch[i], prototypes, threshold)
        expected = -1 if single_label is None else single_label
        assert preds[i] == expected
        assert np.isclose(dists[i], single_dist)


# ---------------------------------------------------------------------
# compute_zero_day_metrics
# ---------------------------------------------------------------------


def test_zero_day_metrics_perfect_detection_and_no_false_positives():
    zero_day_preds = np.array([-1, -1, -1])
    known_preds = np.array([0, 1, 0, 1])
    m = compute_zero_day_metrics(zero_day_preds, known_preds)
    assert m["zero_day_detection_rate"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_zero_day_metrics_partial_detection_with_false_positives():
    zero_day_preds = np.array([-1, -1, 0, 1])  # 2/4 detected
    known_preds = np.array([0, -1, 1, 1])  # 1/4 false positive
    m = compute_zero_day_metrics(zero_day_preds, known_preds)
    assert np.isclose(m["zero_day_detection_rate"], 0.5)
    assert np.isclose(m["false_positive_rate"], 0.25)
