"""Temporal smoothness metrics for trajectory datasets.

Both functions accept batched ``(N, T, D)`` or single-sample ``(T, D)``
input arrays.  Lower values indicate smoother trajectories.
"""

from __future__ import annotations

import numpy as np


def temporal_smoothness(
    x: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Average second-order temporal difference (acceleration) norm.

    Computed as::

        S(x) = mean_t  ||x_{t+1} - 2 x_t + x_{t-1}||_2

    Parameters
    ----------
    x   : Array of shape ``(N, T, D)`` or ``(T, D)``.
    eps : Small constant added to the denominator for numerical stability.

    Returns
    -------
    mean_smoothness : Dataset-level mean (scalar float).
    per_sample      : Array of shape ``(N,)`` with per-sample values.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis]
    if x.ndim != 3:
        raise ValueError(f"x must have shape (N, T, D) or (T, D), got {x.shape}")

    if x.shape[1] < 3:
        per_sample = np.zeros(x.shape[0], dtype=np.float32)
        return float(np.mean(per_sample)), per_sample

    accel = x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]  # (N, T-2, D)
    per_timestep = np.linalg.norm(accel, axis=2)                 # (N, T-2)
    per_sample = per_timestep.sum(axis=1) / (x.shape[1] - 2 + eps)
    return float(np.mean(per_sample)), per_sample


def temporal_variation_smoothness(
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Fused-lasso-inspired smoothness of the perturbation signal.

    Measures the total variation of the perturbation ``r = x_adv - x_orig``
    along the time axis::

        TV(r) = sum_t  sum_d  |r_{t+1,d} - r_{t,d}|  / (T - 1)

    Parameters
    ----------
    x_orig : Original trajectories of shape ``(N, T, D)`` or ``(T, D)``.
    x_adv  : Adversarial trajectories of the same shape.
    eps    : Small constant for numerical stability.

    Returns
    -------
    mean_smoothness : Dataset-level mean (scalar float).
    per_sample      : Array of shape ``(N,)`` with per-sample values.
    """
    x_orig = np.asarray(x_orig, dtype=np.float32)
    x_adv = np.asarray(x_adv, dtype=np.float32)

    if x_orig.ndim == 2:
        x_orig = x_orig[np.newaxis]
        x_adv = x_adv[np.newaxis]

    if x_orig.shape != x_adv.shape:
        raise ValueError(
            f"x_orig and x_adv must have the same shape, "
            f"got {x_orig.shape} and {x_adv.shape}"
        )

    r = x_adv - x_orig                                           # (N, T, D)
    diff = r[:, 1:, :] - r[:, :-1, :]                           # (N, T-1, D)
    per_sample = np.abs(diff).sum(axis=(1, 2)) / (r.shape[1] - 1 + eps)
    return float(np.mean(per_sample)), per_sample
