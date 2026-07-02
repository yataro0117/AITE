"""Tests for evaluation metrics (trajectory distance, rendering, smoothness)."""

import numpy as np
import pytest

from aite.metrics.rendering import render_trajectory
from aite.metrics.smoothness import temporal_smoothness, temporal_variation_smoothness
from aite.metrics.trajectory import l2_trajectory, linf_trajectory


def _sample_trajectory(T: int = 20) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)


def test_render_trajectory_output_shape() -> None:
    """render_trajectory should return an array of shape (img_size, img_size)."""
    img = render_trajectory(_sample_trajectory(), img_size=64)
    assert img.shape == (64, 64)


def test_render_trajectory_value_range() -> None:
    """render_trajectory output values should be in [0, 1]."""
    img = render_trajectory(_sample_trajectory(), img_size=48)
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_render_trajectory_dtype() -> None:
    """render_trajectory should return a float32 array."""
    img = render_trajectory(_sample_trajectory(), img_size=32)
    assert img.dtype == np.float32


def test_l2_trajectory_identical() -> None:
    """L2 between identical trajectories should be 0."""
    x = _sample_trajectory()
    assert l2_trajectory(x, x) == pytest.approx(0.0)


def test_l2_trajectory_shape_mismatch() -> None:
    """l2_trajectory should raise ValueError for mismatched shapes."""
    with pytest.raises(ValueError):
        l2_trajectory(_sample_trajectory(10), _sample_trajectory(11))


def test_linf_trajectory_identical() -> None:
    """L-inf between identical trajectories should be 0."""
    x = _sample_trajectory()
    assert linf_trajectory(x, x) == pytest.approx(0.0)


def test_linf_trajectory_value() -> None:
    """L-inf should equal the maximum absolute coordinate difference."""
    x = _sample_trajectory(5)
    adv = x.copy()
    adv[2, 0] += 0.3
    adv[3, 1] -= 0.1
    assert linf_trajectory(x, adv) == pytest.approx(0.3, abs=1e-6)


def test_temporal_smoothness_single_sample() -> None:
    """temporal_smoothness should accept a single (T, D) array."""
    mean, per_sample = temporal_smoothness(_sample_trajectory())
    assert isinstance(mean, float)
    assert per_sample.shape == (1,)


def test_temporal_smoothness_batch() -> None:
    """temporal_smoothness should accept a batch (N, T, D) array."""
    batch = np.stack([_sample_trajectory() for _ in range(4)], axis=0)
    mean, per_sample = temporal_smoothness(batch)
    assert per_sample.shape == (4,)
    assert mean == pytest.approx(float(per_sample.mean()), rel=1e-5)


def test_temporal_variation_smoothness_shape_mismatch() -> None:
    """temporal_variation_smoothness should raise on mismatched shapes."""
    with pytest.raises(ValueError):
        temporal_variation_smoothness(
            _sample_trajectory(10)[np.newaxis], _sample_trajectory(11)[np.newaxis]
        )


def test_temporal_variation_smoothness_zero_perturbation() -> None:
    """Zero perturbation should give smoothness 0."""
    x = _sample_trajectory()
    mean, per_sample = temporal_variation_smoothness(x, x.copy())
    assert mean == pytest.approx(0.0)
    assert np.allclose(per_sample, 0.0)
