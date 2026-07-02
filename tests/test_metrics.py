"""Tests for evaluation metrics (trajectory distance, rendering, smoothness)."""

from aite.metrics.rendering import render_trajectory
from aite.metrics.smoothness import temporal_smoothness, temporal_variation_smoothness
from aite.metrics.trajectory import l2_trajectory, linf_trajectory


def test_render_trajectory_output_shape() -> None:
    """render_trajectory should return an array of shape (img_size, img_size)."""
    pass


def test_render_trajectory_value_range() -> None:
    """render_trajectory output values should be in [0, 1]."""
    pass


def test_render_trajectory_dtype() -> None:
    """render_trajectory should return a float32 array."""
    pass


def test_l2_trajectory_identical() -> None:
    """L2 between identical trajectories should be 0."""
    pass


def test_l2_trajectory_shape_mismatch() -> None:
    """l2_trajectory should raise ValueError for mismatched shapes."""
    pass


def test_linf_trajectory_identical() -> None:
    """L-inf between identical trajectories should be 0."""
    pass


def test_temporal_smoothness_single_sample() -> None:
    """temporal_smoothness should accept a single (T, D) array."""
    pass


def test_temporal_smoothness_batch() -> None:
    """temporal_smoothness should accept a batch (N, T, D) array."""
    pass


def test_temporal_variation_smoothness_shape_mismatch() -> None:
    """temporal_variation_smoothness should raise on mismatched shapes."""
    pass


def test_temporal_variation_smoothness_zero_perturbation() -> None:
    """Zero perturbation should give smoothness 0."""
    pass
