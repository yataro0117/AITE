"""Low-level temporal editing primitives for 2-D pen-stroke trajectories.

Both functions operate on raw numpy arrays of shape ``(T, 2)`` and do *not*
restore sequence length — callers are responsible for resampling via
:func:`aite.data.unipen.interpolate_add` or
:func:`aite.data.unipen.interpolate_del` when fixed-length input is required.
"""

from __future__ import annotations

import numpy as np


def insert_midpoint(x: np.ndarray, interval_idx: int) -> np.ndarray:
    """Insert the midpoint of interval ``[interval_idx, interval_idx+1]``.

    Parameters
    ----------
    x            : Trajectory of shape ``(T, 2)``.
    interval_idx : Index of the left point of the interval where the midpoint
                   will be inserted.  Must satisfy ``0 <= interval_idx < T-1``.

    Returns
    -------
    New trajectory of shape ``(T+1, 2)`` with the midpoint inserted at
    position ``interval_idx + 1``.

    Raises
    ------
    ValueError : If ``x`` is not 2-D with shape ``(T, 2)`` or ``interval_idx``
                 is out of range.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x with shape (T, 2), got {x.shape}")
    T = x.shape[0]
    if T < 2:
        raise ValueError(f"Trajectory must have at least 2 points, got {T}")
    if not (0 <= interval_idx < T - 1):
        raise ValueError(
            f"interval_idx must be in [0, {T - 2}], got {interval_idx}"
        )
    mid = ((x[interval_idx] + x[interval_idx + 1]) / 2.0).astype(np.float32)
    return np.insert(x, interval_idx + 1, mid, axis=0)


def delete_point(x: np.ndarray, point_idx: int) -> np.ndarray:
    """Delete the point at ``point_idx`` from the trajectory.

    Parameters
    ----------
    x          : Trajectory of shape ``(T, 2)``.
    point_idx  : Index of the point to remove.  Must satisfy
                 ``0 <= point_idx < T``.

    Returns
    -------
    New trajectory of shape ``(T-1, 2)`` with the specified point removed.

    Raises
    ------
    ValueError : If ``x`` is not 2-D with shape ``(T, 2)`` or ``point_idx``
                 is out of range.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x with shape (T, 2), got {x.shape}")
    T = x.shape[0]
    if T < 2:
        raise ValueError(
            f"Cannot delete from a trajectory with fewer than 2 points, got {T}"
        )
    if not (0 <= point_idx < T):
        raise ValueError(f"point_idx must be in [0, {T - 1}], got {point_idx}")
    return np.delete(x, point_idx, axis=0)
