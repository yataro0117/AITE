"""Gradient-based baseline adversarial attacks for padded trajectory batches.

All public functions share the signature::

    attack_fn(model, x, y, **kwargs) -> torch.Tensor

where

* ``model`` is a callable mapping ``x[B, T, C]`` to logits ``[B, K]``.
* ``x`` is a ``FloatTensor[B, T, C]`` (values in ``[-1, 1]``).
* ``y`` is a ``LongTensor[B]`` of ground-truth class indices.

For CASIA's variable-length batches, pass ``valid_mask=batch["mask"]`` to
zero out gradients / perturbations at padded positions.

Attacks
-------
fgsm          : Fast Gradient Sign Method
bim           : Basic Iterative Method (= PGD without random init)
pgd           : Projected Gradient Descent
cw_l2         : Carlini-Wagner L2
mi_fgsm       : Momentum Iterative FGSM
ni_fgsm       : Nesterov Iterative FGSM
ti_mi_fgsm_1d : Translation-Invariant MI-FGSM (1-D temporal smoothing)
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared internal utilities
# ---------------------------------------------------------------------------

def _autocast_disabled(device: torch.device):
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except Exception:
        return nullcontext()


def _eval_mode(model) -> None:
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()


def _expand_mask(
    valid_mask: Optional[torch.Tensor], ref: torch.Tensor
) -> Optional[torch.Tensor]:
    """Expand a ``[B, T]`` mask to ``ref``'s shape for elementwise ops."""
    if valid_mask is None:
        return None
    mask = valid_mask.to(device=ref.device)
    while mask.ndim < ref.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(ref).to(dtype=ref.dtype)


def _apply_mask(
    x: torch.Tensor, valid_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    m = _expand_mask(valid_mask, x)
    return x if m is None else x * m


def _restore_padding(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    m = _expand_mask(valid_mask, x_adv)
    if m is None:
        return x_adv
    return x_adv * m + x_orig * (1.0 - m)


def _compute_gradient(
    model_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    valid_mask: Optional[torch.Tensor] = None,
    targeted: bool = False,
) -> torch.Tensor:
    """Compute the input gradient of the loss w.r.t. ``x``."""
    x_var = x.detach().clone().to(dtype=torch.float32)
    x_var.requires_grad_(True)
    y_in = y.detach().to(device=x_var.device, dtype=torch.long)

    with _autocast_disabled(x_var.device):
        logits = model_fn(x_var)
        loss = loss_fn(logits, y_in)
        if loss.ndim != 0:
            loss = loss.sum()
        if targeted:
            loss = -loss

    grad = torch.autograd.grad(loss, x_var)[0]
    return _apply_mask(grad.detach(), valid_mask)


def _clip_eta(eta: torch.Tensor, ord, eps: float) -> torch.Tensor:
    """Project ``eta`` onto an Lp ball of radius ``eps``."""
    if ord == np.inf:
        return torch.clamp(eta, -eps, eps)
    flat = eta.reshape(eta.shape[0], -1)
    if ord == 2:
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(norm), torch.full_like(norm, eps) / norm)
        return (flat * scale).reshape_as(eta)
    raise ValueError(f"Unsupported norm: {ord}")


def _optimize_linear(grad: torch.Tensor, eps: float, ord) -> torch.Tensor:
    """Compute the attack step direction."""
    flat = grad.reshape(grad.shape[0], -1)
    if ord == np.inf:
        return (eps * flat.sign()).reshape_as(grad)
    if ord == 2:
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (eps * flat / norm).reshape_as(grad)
    raise ValueError(f"Unsupported norm: {ord}")


def _normalize_by_mean_abs(
    grad: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    abs_grad = grad.abs()
    if valid_mask is None:
        denom = abs_grad.reshape(abs_grad.shape[0], -1).mean(dim=1)
    else:
        m = _expand_mask(valid_mask, grad)
        assert m is not None
        flat_abs = (abs_grad * m).reshape(abs_grad.shape[0], -1).sum(dim=1)
        valid_count = m.reshape(m.shape[0], -1).sum(dim=1).clamp_min(1.0)
        denom = flat_abs / valid_count
    denom = denom.clamp_min(1e-12)
    return grad / denom.view(denom.shape[0], *([1] * (grad.ndim - 1)))


def _project_and_clip(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    eps: float,
    valid_mask: Optional[torch.Tensor],
    clip_min: float,
    clip_max: float,
    ord,
) -> torch.Tensor:
    eta = _clip_eta(x_adv - x_orig, ord, eps)
    eta = _apply_mask(eta.to(device=x_orig.device, dtype=x_orig.dtype), valid_mask)
    x_adv = torch.clamp(x_orig + eta, clip_min, clip_max)
    return _restore_padding(x_adv, x_orig, valid_mask)


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    valid_mask: Optional[torch.Tensor] = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    loss_fn: Optional[Callable] = None,
    ord=np.inf,
) -> torch.Tensor:
    """Fast Gradient Sign Method.

    Parameters
    ----------
    model     : Callable ``x[B, T, C] -> logits[B, K]``.
    x         : Input tensor ``[B, T, C]``.
    y         : Ground-truth labels ``[B]``.
    eps       : L-inf perturbation budget.
    valid_mask: Optional validity mask ``[B, T]`` (for CASIA padded batches).
    clip_min  : Lower input bound.
    clip_max  : Upper input bound.
    loss_fn   : Loss function; defaults to cross-entropy.
    ord       : Norm order (only ``np.inf`` supported).

    Returns
    -------
    Adversarial tensor same shape as ``x``.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy
    _eval_mode(model)

    grad = _compute_gradient(model, x, y, loss_fn, valid_mask)
    step = _optimize_linear(grad, eps, ord)
    step = _apply_mask(step.to(device=x.device, dtype=x.dtype), valid_mask)
    x_adv = torch.clamp(x + step, clip_min, clip_max)
    return _restore_padding(x_adv, x, valid_mask).detach()


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    eps_iter: float,
    nb_iter: int,
    valid_mask: Optional[torch.Tensor] = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    loss_fn: Optional[Callable] = None,
    ord=np.inf,
    rand_init: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent.

    Parameters
    ----------
    model     : Callable ``x[B, T, C] -> logits[B, K]``.
    x         : Input tensor ``[B, T, C]``.
    y         : Ground-truth labels ``[B]``.
    eps       : L-inf perturbation budget.
    eps_iter  : Step size per iteration.
    nb_iter   : Number of PGD steps.
    valid_mask: Optional validity mask ``[B, T]``.
    clip_min  : Lower input bound.
    clip_max  : Upper input bound.
    loss_fn   : Loss function; defaults to cross-entropy.
    ord       : Norm order (only ``np.inf`` supported).
    rand_init : Whether to start from a random point in the eps-ball.

    Returns
    -------
    Adversarial tensor same shape as ``x``.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy
    _eval_mode(model)
    x_orig = x.detach()

    if rand_init:
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        delta = _apply_mask(delta, valid_mask)
        x_adv = torch.clamp(x_orig + delta, clip_min, clip_max)
        x_adv = _restore_padding(x_adv, x_orig, valid_mask)
    else:
        x_adv = x_orig.clone()

    for _ in range(nb_iter):
        grad = _compute_gradient(model, x_adv, y, loss_fn, valid_mask)
        step = _optimize_linear(grad, eps_iter, ord)
        step = _apply_mask(step.to(device=x.device, dtype=x.dtype), valid_mask)
        x_adv = _project_and_clip(
            x_adv + step, x_orig, eps, valid_mask, clip_min, clip_max, ord
        )

    return x_adv.detach()


# ---------------------------------------------------------------------------
# BIM (= PGD without rand_init)
# ---------------------------------------------------------------------------

def bim(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    eps_iter: float,
    nb_iter: int,
    valid_mask: Optional[torch.Tensor] = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    loss_fn: Optional[Callable] = None,
    ord=np.inf,
) -> torch.Tensor:
    """Basic Iterative Method (BIM = PGD without random initialization).

    Parameters match :func:`pgd` except ``rand_init`` is always ``False``.
    """
    return pgd(
        model=model, x=x, y=y, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter,
        valid_mask=valid_mask, clip_min=clip_min, clip_max=clip_max,
        loss_fn=loss_fn, ord=ord, rand_init=False,
    )


# ---------------------------------------------------------------------------
# CW-L2
# ---------------------------------------------------------------------------

def _to_tanh(x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    x_s = (x - clip_min) / (clip_max - clip_min)
    return torch.atanh(torch.clamp(x_s, 0.0, 1.0) * 2.0 - 1.0) * 0.999999


def _from_tanh(w: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    return ((torch.tanh(w) + 1.0) / 2.0) * (clip_max - clip_min) + clip_min


def _cw_margin(
    logits: torch.Tensor,
    y: torch.Tensor,
    confidence: float,
    targeted: bool,
) -> torch.Tensor:
    y = y.to(device=logits.device, dtype=torch.long)
    real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(y, logits.shape[1]).bool()
    other = logits.masked_fill(mask, torch.finfo(logits.dtype).min).max(dim=1).values
    if targeted:
        return (other - real + confidence).clamp_min(0.0)
    return (real - other + confidence).clamp_min(0.0)


def cw_l2(
    model,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    targeted: bool = False,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    binary_search_steps: int = 5,
    max_iterations: int = 200,
    abort_early: bool = True,
    confidence: float = 0.0,
    initial_const: float = 1e-2,
    learning_rate: float = 1e-2,
) -> torch.Tensor:
    """Carlini-Wagner L2 attack.

    Parameters
    ----------
    model              : Callable ``x[B, T, C] -> logits[B, K]``.
    x                  : Input tensor ``[B, T, C]``.
    y                  : Ground-truth labels ``[B]``; inferred if ``None``.
    valid_mask         : Optional validity mask ``[B, T]``.
    targeted           : Run targeted attack.
    clip_min / clip_max: Input bounds.
    binary_search_steps: Outer binary search over the trade-off constant.
    max_iterations     : Inner Adam iterations per binary search step.
    abort_early        : Stop inner loop if loss stagnates.
    confidence         : CW confidence margin.
    initial_const      : Initial value of the trade-off constant.
    learning_rate      : Adam learning rate for the inner optimization.

    Returns
    -------
    Adversarial tensor same shape as ``x``.
    """
    _eval_mode(model)
    x_orig = x.detach().clone()
    B = x_orig.shape[0]

    with _autocast_disabled(x_orig.device):
        if y is None:
            logits_init = model(x_orig)
            y_used = logits_init.argmax(dim=1)
        else:
            y_used = y.detach().clone().to(device=x_orig.device, dtype=torch.long)

    x_tanh = _to_tanh(x_orig, clip_min, clip_max)
    lb = torch.zeros(B, device=x_orig.device, dtype=x_orig.dtype)
    ub = torch.full((B,), float("inf"), device=x_orig.device, dtype=x_orig.dtype)
    const = torch.full((B,), float(initial_const), device=x_orig.device, dtype=x_orig.dtype)

    best_l2 = torch.full((B,), float("inf"), device=x_orig.device, dtype=x_orig.dtype)
    best_score = torch.full((B,), -1, device=x_orig.device, dtype=torch.long)
    best_attack = x_orig.clone()
    check_interval = max(max_iterations // 10, 1)

    for outer in range(binary_search_steps):
        if binary_search_steps >= 10 and outer == binary_search_steps - 1:
            finite_ub = torch.isfinite(ub)
            const = torch.where(finite_ub, ub, const)

        modifier = torch.zeros_like(x_tanh, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)
        successful_this_outer = torch.zeros(B, device=x_orig.device, dtype=torch.bool)
        prev_loss: Optional[float] = None

        for it in range(max_iterations):
            optimizer.zero_grad()
            with _autocast_disabled(x_orig.device):
                mod_masked = _apply_mask(modifier, valid_mask)
                x_adv = _restore_padding(
                    _from_tanh(x_tanh + mod_masked, clip_min, clip_max),
                    x_orig, valid_mask,
                )
                logits = model(x_adv)
                delta = _apply_mask(x_adv - x_orig, valid_mask)
                l2 = delta.reshape(B, -1).pow(2).sum(dim=1)
                f = _cw_margin(logits, y_used, confidence, targeted)
                total = l2.sum() + (const * f).sum()

            total.backward()
            if modifier.grad is not None:
                modifier.grad.data = _apply_mask(modifier.grad.data, valid_mask)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                if targeted:
                    success = pred.eq(y_used)
                else:
                    success = pred.ne(y_used)
                successful_this_outer |= success

                improved = success & (l2 < best_l2)
                if improved.any():
                    best_l2 = torch.where(improved, l2.detach(), best_l2)
                    best_score = torch.where(improved, pred.detach(), best_score)
                    sm = improved.view(B, 1, 1).expand_as(best_attack)
                    best_attack = torch.where(sm, x_adv.detach(), best_attack)

                if abort_early and (it + 1) % check_interval == 0:
                    loss_val = float(total.detach().item())
                    if prev_loss is not None and loss_val > prev_loss * 0.9999:
                        break
                    prev_loss = loss_val

        # Update binary search bounds.
        ub = torch.where(successful_this_outer, torch.minimum(ub, const), ub)
        lb = torch.where(successful_this_outer, lb, torch.maximum(lb, const))
        finite_ub = torch.isfinite(ub)
        mid = (lb + ub) / 2.0
        const = torch.where(
            successful_this_outer, mid, torch.where(finite_ub, mid, const * 10.0)
        )

    best_attack = _restore_padding(best_attack, x_orig, valid_mask)
    return best_attack.to(device=x.device, dtype=x.dtype).detach()


# ---------------------------------------------------------------------------
# MI-FGSM
# ---------------------------------------------------------------------------

def mi_fgsm(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    eps_iter: float,
    nb_iter: int,
    valid_mask: Optional[torch.Tensor] = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    loss_fn: Optional[Callable] = None,
    ord=np.inf,
    decay_factor: float = 1.0,
    targeted: bool = False,
) -> torch.Tensor:
    """Momentum Iterative FGSM (MI-FGSM).

    Parameters
    ----------
    model        : Callable ``x[B, T, C] -> logits[B, K]``.
    x            : Input tensor ``[B, T, C]``.
    y            : Ground-truth labels ``[B]``.
    eps          : L-inf perturbation budget.
    eps_iter     : Step size per iteration.
    nb_iter      : Number of iterations.
    valid_mask   : Optional validity mask ``[B, T]``.
    clip_min     : Lower input bound.
    clip_max     : Upper input bound.
    loss_fn      : Loss function; defaults to cross-entropy.
    ord          : Norm order (only ``np.inf`` supported).
    decay_factor : Momentum decay factor (``mu`` in the paper).
    targeted     : Run targeted attack.

    Returns
    -------
    Adversarial tensor same shape as ``x``.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy
    _eval_mode(model)
    x_orig = x.detach()
    x_adv = x_orig.clone()
    momentum = torch.zeros_like(x_orig, dtype=torch.float32)

    for _ in range(nb_iter):
        grad = _compute_gradient(model, x_adv, y, loss_fn, valid_mask, targeted)
        grad = _normalize_by_mean_abs(grad, valid_mask)
        momentum = decay_factor * momentum + grad
        momentum = _apply_mask(momentum, valid_mask)
        step = _apply_mask(
            (eps_iter * momentum.sign()).to(device=x.device, dtype=x.dtype), valid_mask
        )
        x_adv = _project_and_clip(
            x_adv + step, x_orig, eps, valid_mask, clip_min, clip_max, ord
        )

    return x_adv.detach()


# ---------------------------------------------------------------------------
# NI-FGSM
# ---------------------------------------------------------------------------

def ni_fgsm(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    eps_iter: float,
    nb_iter: int,
    valid_mask: Optional[torch.Tensor] = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    loss_fn: Optional[Callable] = None,
    ord=np.inf,
    decay_factor: float = 1.0,
    targeted: bool = False,
) -> torch.Tensor:
    """Nesterov Iterative FGSM (NI-FGSM).

    Parameters match :func:`mi_fgsm`.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy
    _eval_mode(model)
    x_orig = x.detach()
    x_adv = x_orig.clone()
    momentum = torch.zeros_like(x_orig, dtype=torch.float32)

    for _ in range(nb_iter):
        look_ahead = x_adv + (eps_iter * decay_factor) * momentum.to(dtype=x.dtype)
        look_ahead = _project_and_clip(
            look_ahead, x_orig, eps, valid_mask, clip_min, clip_max, ord
        )
        grad = _compute_gradient(model, look_ahead, y, loss_fn, valid_mask, targeted)
        grad = _normalize_by_mean_abs(grad, valid_mask)
        momentum = decay_factor * momentum + grad
        momentum = _apply_mask(momentum, valid_mask)
        step = _apply_mask(
            (eps_iter * momentum.sign()).to(device=x.device, dtype=x.dtype), valid_mask
        )
        x_adv = _project_and_clip(
            x_adv + step, x_orig, eps, valid_mask, clip_min, clip_max, ord
        )

    return x_adv.detach()


# ---------------------------------------------------------------------------
# 1D TI-MI-FGSM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """Build a normalized 1-D Gaussian kernel."""
    pos = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-(pos ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum().clamp_min(1e-12)


def _temporal_smooth_grad(
    grad: torch.Tensor,
    kernel_size: int,
    sigma: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Smooth gradients along the time axis with a per-channel Gaussian kernel."""
    if grad.ndim != 3:
        raise ValueError(f"Expected grad [B, T, C], got {grad.shape}")
    channels = grad.shape[-1]
    kernel = _gaussian_kernel_1d(kernel_size, sigma, grad.device, grad.dtype)
    weight = kernel.view(1, 1, kernel_size).expand(channels, 1, kernel_size).contiguous()
    g = grad.transpose(1, 2)  # [B, C, T]
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    g = F.pad(g, (left, right))
    smoothed = F.conv1d(g, weight, groups=channels).transpose(1, 2)
    return _apply_mask(smoothed, valid_mask)


def ti_mi_fgsm_1d(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    eps_iter: float,
    nb_iter: int,
    valid_mask: Optional[torch.Tensor] = None,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    loss_fn: Optional[Callable] = None,
    ord=np.inf,
    decay_factor: float = 1.0,
    kernel_size: int = 7,
    sigma: float = 3.0,
    targeted: bool = False,
) -> torch.Tensor:
    """Translation-Invariant MI-FGSM with 1-D temporal gradient smoothing.

    Parameters
    ----------
    model        : Callable ``x[B, T, C] -> logits[B, K]``.
    x            : Input tensor ``[B, T, C]``.
    y            : Ground-truth labels ``[B]``.
    eps          : L-inf perturbation budget.
    eps_iter     : Step size per iteration.
    nb_iter      : Number of iterations.
    valid_mask   : Optional validity mask ``[B, T]``.
    clip_min     : Lower input bound.
    clip_max     : Upper input bound.
    loss_fn      : Loss function; defaults to cross-entropy.
    ord          : Norm order (only ``np.inf`` supported).
    decay_factor : Momentum decay factor.
    kernel_size  : Size of the temporal Gaussian smoothing kernel.
    sigma        : Standard deviation of the Gaussian kernel.
    targeted     : Run targeted attack.

    Returns
    -------
    Adversarial tensor same shape as ``x``.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy
    _eval_mode(model)
    x_orig = x.detach()
    x_adv = x_orig.clone()
    momentum = torch.zeros_like(x_orig, dtype=torch.float32)

    for _ in range(nb_iter):
        grad = _compute_gradient(model, x_adv, y, loss_fn, valid_mask, targeted)
        grad = _temporal_smooth_grad(grad, kernel_size, sigma, valid_mask)
        grad = _normalize_by_mean_abs(grad, valid_mask)
        momentum = decay_factor * momentum + grad
        momentum = _apply_mask(momentum, valid_mask)
        step = _apply_mask(
            (eps_iter * momentum.sign()).to(device=x.device, dtype=x.dtype), valid_mask
        )
        x_adv = _project_and_clip(
            x_adv + step, x_orig, eps, valid_mask, clip_min, clip_max, ord
        )

    return x_adv.detach()
