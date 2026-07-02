"""AITE: Adversarial Iterative Temporal Editing.

Core algorithm
--------------
Each iteration:
  1. Compute a salience map ``cam[T]`` via ``salience_fn``.
  2. Delete the ``n_del`` highest-salience interior points.
  3. Insert midpoints at the ``n_add`` lowest-salience intervals.
  4. (Unipen only) Resample back to the fixed target length.
  5. Repeat until misclassification or ``max_iter`` is reached.

Public API
----------
run_aite     : Dataset-agnostic core; accepts a generic ``salience_fn``.
aite_unipen  : Thin wrapper for fixed-length Unipen models.
aite_casia   : Thin wrapper for variable-length CASIA models.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from aite.salience.gradcam import compute_gradcam


# ---------------------------------------------------------------------------
# Internal utilities (shared by both wrappers)
# ---------------------------------------------------------------------------

def _autocast_disabled(device: torch.device):
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except Exception:
        return nullcontext()


def _get_model_device(model: nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    if param is not None:
        return param.device
    buf = next(model.buffers(), None)
    if buf is not None:
        return buf.device
    return torch.device("cpu")


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected trajectory with shape [T, 2], got {arr.shape}")
    return np.asarray(arr, dtype=np.float32).copy()


def _align_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample trajectory to exactly ``target_len`` timesteps (linear interp)."""
    T = x.shape[0]
    if T == target_len:
        return x
    old = np.linspace(0, target_len - 1, num=T, dtype=np.float32)
    new = np.arange(target_len, dtype=np.float32)
    return np.stack(
        [np.interp(new, old, x[:, d]) for d in range(x.shape[1])], axis=1
    ).astype(np.float32)


def _interior_candidates(length: int) -> np.ndarray:
    """Return deletable indices, excluding the first and last points."""
    if length <= 2:
        return np.empty((0,), dtype=np.int64)
    return np.arange(1, length - 1, dtype=np.int64)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

SalienceFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""Type alias: ``salience_fn(x) -> (cam[T], logits[K])``."""


def run_aite(
    x: np.ndarray,
    y: int,
    salience_fn: SalienceFn,
    max_iter: int = 100,
    n_del: int = 1,
    n_add: int = 1,
    fixed_length: Optional[int] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Dataset-agnostic AITE core loop.

    Parameters
    ----------
    x            : Original trajectory ``[T, 2]``.
    y            : True class index.
    salience_fn  : Callable returning ``(cam[T], logits[K])`` for the current
                   trajectory.  ``cam`` is a signed saliency map in ``[-1, 1]``.
    max_iter     : Maximum edit steps.
    n_del        : Points to delete per step (highest CAM interior points).
    n_add        : Midpoints to insert per step (lowest CAM intervals).
    fixed_length : If not ``None``, resample back to this length after every
                   step (use for Unipen with fixed T; leave ``None`` for CASIA).
    verbose      : Print per-step progress to stdout.

    Returns
    -------
    adv_x   : Adversarial (or last-edited) trajectory ``[T, 2]``.
    history : List of per-step dicts with keys ``step``, ``pred``, ``length``,
              ``del_indices``, ``add_intervals``, ``success``.
    """
    x = _to_numpy(x)
    y = int(y)
    history: list[dict] = []

    for step in range(max_iter):
        cam, logits = salience_fn(x)
        pred = int(np.argmax(logits))

        if pred != y:
            if verbose:
                print(f"[AITE] success before edit at step {step}: pred={pred}")
            history.append({
                "step": step,
                "pred": pred,
                "length": len(x),
                "del_indices": [],
                "add_intervals": [],
                "success": True,
            })
            return x, history

        # ---- deletion: highest CAM interior points ----
        del_idx = np.empty((0,), dtype=np.int64)
        del_candidates = _interior_candidates(len(x))
        if len(del_candidates) > 0 and n_del > 0:
            budget = min(n_del, len(del_candidates))
            ranked = np.argsort(cam[del_candidates])[-budget:]
            del_idx = np.sort(del_candidates[ranked].astype(np.int64))

        keep_idx = np.setdiff1d(np.arange(len(x), dtype=np.int64), del_idx)
        x_after_del = x[keep_idx]

        # ---- insertion: midpoints at lowest CAM intervals ----
        add_intervals = np.empty((0,), dtype=np.int64)
        adjusted_adds: list[int] = []
        if len(x_after_del) > 1 and n_add > 0 and len(cam) > 1:
            budget = min(n_add, len(cam) - 1)
            interval_imp = (cam[:-1] + cam[1:]) / 2.0
            add_intervals = np.argsort(interval_imp)[:budget].astype(np.int64)
            for iv in add_intervals.tolist():
                n_del_before = int(np.sum(del_idx <= iv))
                adj = int(iv - n_del_before)
                if 0 <= adj < len(x_after_del) - 1 and adj not in adjusted_adds:
                    adjusted_adds.append(adj)

        x_next = x_after_del.copy()
        for ai in reversed(sorted(adjusted_adds)):
            mid = ((x_next[ai] + x_next[ai + 1]) / 2.0).astype(np.float32)
            x_next = np.insert(x_next, ai + 1, mid, axis=0)

        if fixed_length is not None:
            x_next = _align_length(x_next, fixed_length)
        x = x_next.astype(np.float32)

        # evaluate after edit
        _, updated_logits = salience_fn(x)
        updated_pred = int(np.argmax(updated_logits))
        success = updated_pred != y

        record = {
            "step": step + 1,
            "pred": updated_pred,
            "length": len(x),
            "del_indices": del_idx.tolist(),
            "add_intervals": add_intervals.tolist(),
            "success": bool(success),
        }
        history.append(record)

        if verbose:
            print(
                f"[AITE] step={step + 1:>4d} pred={updated_pred} "
                f"del={record['del_indices']} add={record['add_intervals']} "
                f"success={success}"
            )

        if success:
            return x, history

    return x, history


# ---------------------------------------------------------------------------
# Unipen wrapper
# ---------------------------------------------------------------------------

def aite_unipen(
    orig_x,
    orig_y: int,
    model: nn.Module,
    max_iter: int = 100,
    n_del: int = 1,
    n_add: int = 1,
    layer_name=None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """AITE for Unipen models (fixed-length, ``forward(x) -> logits``).

    Parameters
    ----------
    orig_x      : Original trajectory ``[T, 2]``.
    orig_y      : True class index.
    model       : Unipen torch model (``forward(x[B, T, 2]) -> logits[B, K]``).
    max_iter    : Maximum edit steps.
    n_del       : Interior points to delete per step.
    n_add       : Midpoints to insert per step.
    layer_name  : Target Conv1d layer for Grad-CAM; ``None`` = last Conv1d.
    verbose     : Print step-by-step progress.

    Returns
    -------
    adv_x   : Adversarial trajectory ``[T, 2]``.
    history : List of per-step dicts.
    """
    model.eval()
    x0 = _to_numpy(orig_x)
    target_len = x0.shape[0]
    device = _get_model_device(model)

    def _salience(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam, logits = compute_gradcam(
            model=model,
            x=x,
            layer_name=layer_name,
            class_idx=None,
            relu=False,
        )
        return cam, logits

    return run_aite(
        x=x0,
        y=orig_y,
        salience_fn=_salience,
        max_iter=max_iter,
        n_del=n_del,
        n_add=n_add,
        fixed_length=target_len,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CASIA wrapper
# ---------------------------------------------------------------------------

def aite_casia(
    orig_x,
    orig_y: int,
    model: nn.Module,
    max_iter: int = 100,
    n_del: int = 1,
    n_add: int = 1,
    layer_name=None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """AITE for CASIA models (variable-length, ``forward(ink, lengths, mask) -> logits``).

    Parameters
    ----------
    orig_x      : Original trajectory ``[T, 2]``.
    orig_y      : True class index.
    model       : CASIA torch model.
    max_iter    : Maximum edit steps.
    n_del       : Interior points to delete per step.
    n_add       : Midpoints to insert per step.
    layer_name  : Target Conv1d layer for Grad-CAM; ``None`` = last Conv1d.
    verbose     : Print step-by-step progress.

    Returns
    -------
    adv_x   : Adversarial trajectory ``[T, 2]`` (variable length).
    history : List of per-step dicts.
    """
    model.eval()
    x0 = _to_numpy(orig_x)

    def _salience(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam, logits = compute_gradcam(
            model=model,
            x=x,
            layer_name=layer_name,
            class_idx=None,
            relu=False,
            casia_model=True,
        )
        return cam, logits

    return run_aite(
        x=x0,
        y=orig_y,
        salience_fn=_salience,
        max_iter=max_iter,
        n_del=n_del,
        n_add=n_add,
        fixed_length=None,  # variable length for CASIA
        verbose=verbose,
    )
