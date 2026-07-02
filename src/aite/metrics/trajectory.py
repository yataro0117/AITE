"""Coordinate-space distance metrics for pairs of trajectories.

These metrics operate directly on the raw ``(T, 2)`` float arrays, not on
rendered images.  For image-space metrics, see :mod:`aite.metrics.rendering`.
"""

from __future__ import annotations

import numpy as np


def l2_trajectory(
    orig: np.ndarray,
    adv: np.ndarray,
) -> float:
    """L2 norm of the perturbation between two equal-length trajectories.

    Parameters
    ----------
    orig : Original trajectory of shape ``(T, 2)``.
    adv  : Adversarial trajectory of shape ``(T, 2)``.

    Returns
    -------
    Scalar L2 distance as a Python float.

    Raises
    ------
    ValueError : If ``orig`` and ``adv`` have different shapes.
    """
    orig = np.asarray(orig, dtype=np.float64)
    adv = np.asarray(adv, dtype=np.float64)
    if orig.shape != adv.shape:
        raise ValueError(
            f"orig and adv must have the same shape, got {orig.shape} vs {adv.shape}"
        )
    return float(np.linalg.norm((adv - orig).ravel()))


def linf_trajectory(
    orig: np.ndarray,
    adv: np.ndarray,
) -> float:
    """L-infinity norm of the perturbation between two equal-length trajectories.

    Parameters
    ----------
    orig : Original trajectory of shape ``(T, 2)``.
    adv  : Adversarial trajectory of shape ``(T, 2)``.

    Returns
    -------
    Scalar L-inf distance as a Python float.

    Raises
    ------
    ValueError : If ``orig`` and ``adv`` have different shapes.
    """
    orig = np.asarray(orig, dtype=np.float64)
    adv = np.asarray(adv, dtype=np.float64)
    if orig.shape != adv.shape:
        raise ValueError(
            f"orig and adv must have the same shape, got {orig.shape} vs {adv.shape}"
        )
    return float(np.max(np.abs(adv - orig)))
