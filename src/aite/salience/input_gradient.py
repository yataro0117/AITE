"""Input-gradient salience for trajectory models.

Computes ``d score[class_idx] / d x`` for a single trajectory sample and
aggregates the per-coordinate gradients into a 1-D salience map over time.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _autocast_disabled(device: torch.device):
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except Exception:
        return nullcontext()


def _get_model_device(model: nn.Module) -> torch.device:
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    b = next(model.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_input_gradient(
    model: nn.Module,
    x,
    class_idx: Optional[int] = None,
    abs_value: bool = True,
    reduce: str = "norm",
    casia_model: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the input-gradient salience map for a single trajectory.

    The gradient ``d logits[class_idx] / d x`` has shape ``[T, 2]``.
    It is reduced to a 1-D salience map ``[T]`` via ``reduce``.

    Parameters
    ----------
    model       : PyTorch model.
    x           : Single trajectory ``[T, 2]`` (array or tensor).
    class_idx   : Class whose logit is differentiated.  ``None`` = predicted.
    abs_value   : Take the absolute value before reducing (default ``True``).
    reduce      : How to collapse the 2-D gradient to 1-D.  Options:

                  * ``'norm'``  – L2 norm across the coordinate axis (default).
                  * ``'mean'``  – arithmetic mean across coordinates.
                  * ``'sum'``   – sum across coordinates.
    casia_model : Set to ``True`` for models that accept
                  ``(ink, lengths, mask)`` instead of a plain tensor.

    Returns
    -------
    salience  : 1-D array of shape ``[T]``, normalized to ``[0, 1]``
                (or ``[-1, 1]`` when ``abs_value=False``).
    logits_np : Logits array of shape ``[num_classes]``.
    """
    model.eval()
    device = _get_model_device(model)

    if torch.is_tensor(x):
        x_arr = x.detach().cpu().numpy().astype(np.float32)
    else:
        x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected x with shape [T, 2], got {x_arr.shape}")
    T = x_arr.shape[0]

    x_t = torch.as_tensor(x_arr, device=device, dtype=torch.float32).unsqueeze(0)
    x_t.requires_grad_(True)

    if casia_model:
        lengths = torch.tensor([T], device=device, dtype=torch.long)
        mask = torch.ones((1, T), device=device, dtype=torch.bool)

    model.zero_grad(set_to_none=True)
    with _autocast_disabled(device):
        if casia_model:
            logits = model(x_t, lengths, mask)
        else:
            logits = model(x_t)

    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())

    logits[0, class_idx].backward()

    if x_t.grad is None:
        raise RuntimeError("Gradient w.r.t. x is None.")

    grad = x_t.grad[0].detach().cpu().numpy()  # [T, 2]

    if abs_value:
        grad = np.abs(grad)

    if reduce == "norm":
        salience = np.linalg.norm(grad, axis=1)
    elif reduce == "mean":
        salience = grad.mean(axis=1)
    elif reduce == "sum":
        salience = grad.sum(axis=1)
    else:
        raise ValueError(f"Unknown reduce method '{reduce}'. Use 'norm', 'mean', or 'sum'.")

    max_val = float(np.max(np.abs(salience))) if salience.size > 0 else 0.0
    if max_val > 0.0:
        salience = salience / max_val

    logits_np = logits[0].detach().cpu().numpy().astype(np.float32, copy=False)
    model.zero_grad(set_to_none=True)
    return salience.astype(np.float32), logits_np
