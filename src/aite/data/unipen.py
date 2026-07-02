"""Unipen dataset loading and preprocessing utilities.

Reads the flat-text Unipen format (2*T floats per line) and normalizes
coordinate values to the [-1, 1] range expected by all AITE models.

Data layout expected on disk (relative to a configurable ``data_root``)::

    data/Unipen/
        raw-train-data-{1a,1b,1c}.txt
        raw-test-data-{1a,1b,1c}.txt
        train-label-{1a,1b,1c}.txt
        test-label-{1a,1b,1c}.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_DATASETS = frozenset({"1a", "1b", "1c"})


def load_unipen(
    dataset: str,
    data_root: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalize a Unipen subset.

    Parameters
    ----------
    dataset   : One of ``'1a'``, ``'1b'``, or ``'1c'``.
    data_root : Path to the project root (the directory that contains
                ``data/Unipen/``).  Defaults to three levels above this
                file (i.e. the repo root when installed with ``pip -e .``).

    Returns
    -------
    train_data   : float32 array of shape ``(N_train, T, 2)`` in ``[-1, 1]``.
    train_labels : int array of shape ``(N_train,)``.
    test_data    : float32 array of shape ``(N_test, T, 2)`` in ``[-1, 1]``.
    test_labels  : int array of shape ``(N_test,)``.
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(VALID_DATASETS)}, got '{dataset}'"
        )

    root = _resolve_data_root(data_root)
    unipen_dir = root / "data" / "Unipen"

    train_data = _load_raw_data(unipen_dir / f"raw-train-data-{dataset}.txt")
    test_data = _load_raw_data(unipen_dir / f"raw-test-data-{dataset}.txt")
    train_labels = _load_labels(unipen_dir / f"train-label-{dataset}.txt")
    test_labels = _load_labels(unipen_dir / f"test-label-{dataset}.txt")

    train_data = data_regularization(train_data)
    test_data = data_regularization(test_data)

    return train_data, train_labels, test_data, test_labels


def data_regularization(dataset: np.ndarray) -> np.ndarray:
    """Normalize dataset values to ``[-1, 1]`` using global min/max.

    Parameters
    ----------
    dataset : Array of shape ``(N, T, 2)`` or any shape with numeric values.

    Returns
    -------
    Normalized float32 array of the same shape.
    """
    arr = np.asarray(dataset, dtype=np.float64)
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    span = max_val - min_val
    if span == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (2.0 * (arr - min_val) / span - 1.0).astype(np.float32)


def interpolate_del(data: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Resample trajectory after point deletion back to the original length.

    Takes the subset of ``data`` selected by ``index``, then linearly
    interpolates it back to ``len(data)`` timesteps.

    Parameters
    ----------
    data  : Original trajectory of shape ``(T, 2)``.
    index : Indices of the points to *keep* (i.e. the complement of the
            deleted indices).

    Returns
    -------
    Resampled trajectory of shape ``(T, 2)`` as float32.
    """
    T = int(data.shape[0])
    kept_x = data[index, 0]
    kept_y = data[index, 1]

    target_idx = np.arange(T, dtype=np.float64)
    src_idx = np.linspace(0, T - 1, num=kept_x.shape[0], dtype=np.float64)

    new_x = np.interp(target_idx, src_idx, kept_x)
    new_y = np.interp(target_idx, src_idx, kept_y)
    return np.column_stack((new_x, new_y)).astype(np.float32)


def interpolate_add(series: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a lengthened trajectory back to ``target_length`` timesteps.

    Used after point insertion to restore the fixed sequence length.

    Parameters
    ----------
    series        : Trajectory of shape ``(current_length, 2)``.
    target_length : Desired output length.

    Returns
    -------
    Resampled trajectory of shape ``(target_length, 2)`` as float32.
    """
    current_length = int(len(series))
    src_idx = np.linspace(0, target_length - 1, num=current_length, dtype=np.float64)
    target_idx = np.arange(target_length, dtype=np.float64)

    new_x = np.interp(target_idx, src_idx, series[:, 0])
    new_y = np.interp(target_idx, src_idx, series[:, 1])
    return np.column_stack((new_x, new_y)).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    # src/aite/data/unipen.py -> repo root is three levels up
    return Path(__file__).resolve().parents[3]


def _load_raw_data(path: Path) -> np.ndarray:
    """Read a Unipen flat-text data file into an (N, T, 2) float32 array."""
    arr = np.loadtxt(path, dtype=np.float64)
    T2 = arr.shape[1]
    if T2 % 2 != 0:
        raise ValueError(
            f"Expected an even number of columns in {path}, got {T2}"
        )
    T = T2 // 2
    return arr.reshape(arr.shape[0], T, 2).astype(np.float32)


def _load_labels(path: Path) -> np.ndarray:
    """Read a Unipen label file into an int64 array."""
    labels: list[int] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                labels.append(int(parts[1]))
    return np.array(labels, dtype=np.int64)
