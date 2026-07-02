"""Grad-CAM salience for 1-D Conv trajectory models.

Provides a unified implementation that works for both Unipen models
(``forward(x) -> logits``) and CASIA models
(``forward(ink, lengths, mask) -> logits``).

Public API
----------
find_last_conv1d(model)
    Return the name of the last ``nn.Conv1d`` in a model.

compute_gradcam(model, x, layer_name, class_idx, relu, casia_model)
    Compute a signed Grad-CAM map for a single trajectory.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
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


def _resize_cam(cam: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate 1-D CAM array to ``target_len``."""
    if cam.shape[0] == target_len:
        return cam.astype(np.float32, copy=False)
    if cam.shape[0] == 1:
        return np.full(target_len, float(cam[0]), dtype=np.float32)
    old = np.linspace(0, target_len - 1, num=cam.shape[0], dtype=np.float32)
    new = np.arange(target_len, dtype=np.float32)
    return np.interp(new, old, cam).astype(np.float32, copy=False)


def _resolve_layer(
    model: nn.Module, layer_name
) -> tuple[str, nn.Conv1d]:
    """Resolve ``layer_name`` to a ``(name, nn.Conv1d)`` tuple."""
    named = dict(model.named_modules())
    if layer_name is None:
        layer_name = find_last_conv1d(model)
    if isinstance(layer_name, nn.Conv1d):
        for name, mod in model.named_modules():
            if mod is layer_name:
                return name, layer_name
        raise ValueError("Provided nn.Conv1d is not part of the model.")
    if not isinstance(layer_name, str):
        raise TypeError("layer_name must be None, a str, or an nn.Conv1d instance.")
    if layer_name not in named:
        raise ValueError(f"Layer '{layer_name}' not found in model.")
    mod = named[layer_name]
    if not isinstance(mod, nn.Conv1d):
        raise ValueError(f"Layer '{layer_name}' is not an nn.Conv1d.")
    return layer_name, mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_last_conv1d(model: nn.Module) -> str:
    """Return the name of the last ``nn.Conv1d`` in ``model``.

    Parameters
    ----------
    model : An ``nn.Module`` that contains at least one ``nn.Conv1d``.

    Returns
    -------
    Name string suitable for passing to :func:`compute_gradcam`.

    Raises
    ------
    ValueError : If no ``nn.Conv1d`` is found.
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv1d):
            return name
    raise ValueError("No nn.Conv1d found in model.")


def compute_gradcam(
    model: nn.Module,
    x,
    layer_name=None,
    class_idx: Optional[int] = None,
    relu: bool = False,
    casia_model: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a signed Grad-CAM map for a single trajectory.

    Unified implementation for Unipen (``forward(x) -> logits``) and
    CASIA (``forward(ink, lengths, mask) -> logits``) models.

    Parameters
    ----------
    model       : PyTorch model containing at least one ``nn.Conv1d``.
    x           : Single trajectory array or tensor of shape ``[T, 2]``.
    layer_name  : Name of the target ``nn.Conv1d``, or the module itself, or
                  ``None`` to auto-select the last Conv1d.
    class_idx   : Class index whose score is back-propagated.  ``None`` uses
                  the predicted (argmax) class.
    relu        : If ``True``, apply ReLU to the CAM (standard Grad-CAM).
                  If ``False`` (default), keep the signed map.
    casia_model : Set to ``True`` for CASIA models that accept
                  ``(ink, lengths, mask)`` instead of a plain tensor.

    Returns
    -------
    cam       : Salience map of shape ``[T]``, normalized to ``[-1, 1]``
                (or ``[0, 1]`` when ``relu=True``).
    logits_np : Logits array of shape ``[num_classes]``.
    """
    model.eval()
    device = _get_model_device(model)

    # Prepare input
    if torch.is_tensor(x):
        x_arr = x.detach().cpu().numpy().astype(np.float32)
    else:
        x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected x with shape [T, 2], got {x_arr.shape}")
    input_len = x_arr.shape[0]

    x_tensor = torch.as_tensor(x_arr, device=device, dtype=torch.float32).unsqueeze(0)

    if casia_model:
        T = input_len
        lengths = torch.tensor([T], device=device, dtype=torch.long)
        mask = torch.ones((1, T), device=device, dtype=torch.bool)

    _, target_mod = _resolve_layer(model, layer_name)

    activations: Optional[torch.Tensor] = None
    grads: Optional[torch.Tensor] = None

    def _fwd_hook(_mod, _inp, output):
        nonlocal activations
        activations = output.detach()

        def _bwd_hook(g: torch.Tensor) -> None:
            nonlocal grads
            grads = g.detach()

        output.register_hook(_bwd_hook)

    handle = target_mod.register_forward_hook(_fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with _autocast_disabled(device):
            if casia_model:
                logits = model(x_tensor, lengths, mask)
            else:
                logits = model(x_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        logits[0, class_idx].backward()
    finally:
        handle.remove()

    if activations is None or grads is None:
        raise RuntimeError("Hooks did not capture activations/gradients.")

    # activations / grads: [1, C, T_conv]
    weights = grads.mean(dim=2)                              # [1, C]
    cam = (weights.unsqueeze(-1) * activations).sum(dim=1).squeeze(0)  # [T_conv]
    cam_np = cam.cpu().numpy().astype(np.float32, copy=False)

    if relu:
        cam_np = np.maximum(cam_np, 0.0)

    max_abs = float(np.max(np.abs(cam_np))) if cam_np.size > 0 else 0.0
    if max_abs > 0.0:
        cam_np = cam_np / max_abs

    cam_np = _resize_cam(cam_np, input_len)
    logits_np = logits[0].detach().cpu().numpy().astype(np.float32, copy=False)
    model.zero_grad(set_to_none=True)
    return cam_np, logits_np
