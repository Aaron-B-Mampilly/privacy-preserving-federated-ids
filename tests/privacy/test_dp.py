"""Phase 8: client-level DP mechanism tests (clipping, noise, RDP calibration)."""

import numpy as np
import pytest

from fedpda_ids.privacy.dp import (
    add_gaussian_noise,
    adaptive_clip_threshold,
    calibrate_noise_multiplier,
    calibrate_prototype_noise_std,
    clip_update,
    compute_achieved_epsilon,
    compute_update,
    update_norm,
)


def test_compute_update_is_elementwise_difference():
    new = [np.array([3.0, 4.0]), np.array([[1.0]])]
    old = [np.array([1.0, 1.0]), np.array([[0.5]])]
    update = compute_update(new, old)
    assert np.allclose(update[0], [2.0, 3.0])
    assert np.allclose(update[1], [[0.5]])


def test_update_norm_is_one_norm_across_all_arrays():
    # [3,4] has norm 5; [0,0,12] has norm 12; combined = sqrt(5^2+12^2) = 13
    update = [np.array([3.0, 4.0]), np.array([0.0, 0.0, 12.0])]
    assert np.isclose(update_norm(update), 13.0)


def test_adaptive_clip_threshold_is_median():
    assert np.isclose(adaptive_clip_threshold([1.0, 5.0, 3.0]), 3.0)
    assert np.isclose(adaptive_clip_threshold([1.0, 2.0, 3.0, 4.0]), 2.5)


def test_adaptive_clip_threshold_requires_nonempty():
    with pytest.raises(ValueError):
        adaptive_clip_threshold([])


def test_clip_update_leaves_small_update_unchanged():
    update = [np.array([0.1, 0.0])]
    clipped = clip_update(update, clip_norm=1.0)
    assert np.allclose(clipped[0], update[0])


def test_clip_update_shrinks_large_update_preserving_direction_across_arrays():
    # combined norm = 5 (3-4-5 triangle split across two arrays)
    update = [np.array([3.0, 0.0]), np.array([4.0])]
    clipped = clip_update(update, clip_norm=1.0)
    assert np.isclose(update_norm(clipped), 1.0)
    # same scale factor (1/5) applied to both arrays
    assert np.allclose(clipped[0], [0.6, 0.0])
    assert np.allclose(clipped[1], [0.8])


def test_add_gaussian_noise_zero_multiplier_is_noop():
    summed = [np.array([1.0, 2.0])]
    noised = add_gaussian_noise(summed, noise_multiplier=0.0, clip_norm=1.0, rng=np.random.default_rng(0))
    assert np.allclose(noised[0], summed[0])


def test_add_gaussian_noise_matches_expected_std():
    summed = [np.zeros(20000)]
    noised = add_gaussian_noise(summed, noise_multiplier=2.0, clip_norm=0.5, rng=np.random.default_rng(42))
    expected_std = 2.0 * 0.5
    assert np.isclose(noised[0].std(), expected_std, rtol=0.05)


def test_calibrate_noise_multiplier_infinite_epsilon_is_zero():
    assert calibrate_noise_multiplier(float("inf"), 1e-5, sample_rate=0.2, num_rounds=100) == 0.0


def test_calibrate_noise_multiplier_stronger_privacy_needs_more_noise():
    nm_loose = calibrate_noise_multiplier(8.0, 1e-5, sample_rate=0.2, num_rounds=100)
    nm_tight = calibrate_noise_multiplier(0.5, 1e-5, sample_rate=0.2, num_rounds=100)
    assert nm_tight > nm_loose > 0.0


def test_compute_achieved_epsilon_zero_multiplier_is_infinite():
    assert compute_achieved_epsilon(0.0, sample_rate=0.2, num_rounds=100, delta=1e-5) == float("inf")


def test_compute_achieved_epsilon_round_trips_calibration():
    target_epsilon = 3.0
    delta = 1e-5
    sample_rate = 0.2
    num_rounds = 100
    nm = calibrate_noise_multiplier(target_epsilon, delta, sample_rate, num_rounds)
    achieved = compute_achieved_epsilon(nm, sample_rate, num_rounds, delta)
    assert np.isclose(achieved, target_epsilon, rtol=0.05)


def test_calibrate_prototype_noise_std_infinite_epsilon_is_zero():
    assert calibrate_prototype_noise_std(float("inf"), 1e-5, clip_bound=1.0, num_classes=12) == 0.0


def test_calibrate_prototype_noise_std_scales_with_sqrt_num_classes():
    std_1_class = calibrate_prototype_noise_std(1.0, 1e-5, clip_bound=1.0, num_classes=1)
    std_4_classes = calibrate_prototype_noise_std(1.0, 1e-5, clip_bound=1.0, num_classes=4)
    # same noise_multiplier/sigma-per-release, sensitivity scales by sqrt(num_classes)
    assert np.isclose(std_4_classes / std_1_class, 2.0, rtol=0.01)  # sqrt(4)/sqrt(1) = 2


def test_calibrate_prototype_noise_std_scales_with_clip_bound():
    std_bound_1 = calibrate_prototype_noise_std(1.0, 1e-5, clip_bound=1.0, num_classes=5)
    std_bound_2 = calibrate_prototype_noise_std(1.0, 1e-5, clip_bound=2.0, num_classes=5)
    assert np.isclose(std_bound_2 / std_bound_1, 2.0, rtol=0.01)


def test_calibrate_prototype_noise_std_stronger_privacy_needs_more_noise():
    std_loose = calibrate_prototype_noise_std(8.0, 1e-5, clip_bound=1.0, num_classes=12)
    std_tight = calibrate_prototype_noise_std(0.5, 1e-5, clip_bound=1.0, num_classes=12)
    assert std_tight > std_loose > 0.0
