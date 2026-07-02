"""Tests for temporal editing primitives (insert_midpoint and delete_point)."""

import numpy as np
import pytest

from aite.attacks.temporal_editing import delete_point, insert_midpoint


def _sample_trajectory() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
    )


def test_insert_midpoint_increases_length() -> None:
    """Inserting a midpoint should increase sequence length by exactly 1."""
    x = _sample_trajectory()
    out = insert_midpoint(x, interval_idx=1)
    assert out.shape == (x.shape[0] + 1, 2)


def test_insert_midpoint_preserves_endpoints() -> None:
    """Endpoints must remain unchanged after a midpoint insertion."""
    x = _sample_trajectory()
    out = insert_midpoint(x, interval_idx=1)
    assert np.allclose(out[0], x[0])
    assert np.allclose(out[-1], x[-1])


def test_insert_midpoint_correct_value() -> None:
    """The inserted point should be the arithmetic mean of its neighbors."""
    x = _sample_trajectory()
    out = insert_midpoint(x, interval_idx=1)
    expected = (x[1] + x[2]) / 2.0
    assert np.allclose(out[2], expected)


def test_insert_midpoint_invalid_index() -> None:
    """Out-of-range interval_idx should raise ValueError."""
    x = _sample_trajectory()
    with pytest.raises(ValueError):
        insert_midpoint(x, interval_idx=x.shape[0] - 1)
    with pytest.raises(ValueError):
        insert_midpoint(x, interval_idx=-1)


def test_insert_midpoint_short_trajectory() -> None:
    """A trajectory shorter than 2 points should raise ValueError."""
    x = np.array([[0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        insert_midpoint(x, interval_idx=0)


def test_delete_point_decreases_length() -> None:
    """Deleting a point should decrease sequence length by exactly 1."""
    x = _sample_trajectory()
    out = delete_point(x, point_idx=1)
    assert out.shape == (x.shape[0] - 1, 2)


def test_delete_point_preserves_remaining() -> None:
    """All points other than the deleted one must appear in the output."""
    x = _sample_trajectory()
    out = delete_point(x, point_idx=2)
    expected = np.delete(x, 2, axis=0)
    assert np.allclose(out, expected)


def test_delete_point_invalid_index() -> None:
    """Out-of-range point_idx should raise ValueError."""
    x = _sample_trajectory()
    with pytest.raises(ValueError):
        delete_point(x, point_idx=x.shape[0])
    with pytest.raises(ValueError):
        delete_point(x, point_idx=-1)


def test_delete_point_too_short() -> None:
    """A trajectory with only 1 point should raise ValueError."""
    x = np.array([[0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        delete_point(x, point_idx=0)


def test_roundtrip_shape_invariant() -> None:
    """Inserting then deleting (at the appropriate index) must restore the original shape."""
    x = _sample_trajectory()
    inserted = insert_midpoint(x, interval_idx=1)
    restored = delete_point(inserted, point_idx=2)
    assert restored.shape == x.shape
    assert np.allclose(restored, x)
