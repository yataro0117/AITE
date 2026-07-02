"""Trajectory rasterization for image-space metrics.

Renders a 2-D pen-stroke trajectory to a grayscale float32 image using
matplotlib.  The rendering protocol matches the one used in
``experiments/compare_attacks.py`` and ``CASIA/src/evaluate_adv_samples.py``:

* Figure size ``(img_size/100, img_size/100)`` at DPI 100.
* Axis limits per-sample: ``[min - 0.05, max + 0.05]`` with equal aspect.
* Grayscale inverted so the stroke is bright (``1``) and background is dark (``0``).
"""

from __future__ import annotations

import numpy as np


def render_trajectory(
    xy: np.ndarray,
    img_size: int = 64,
    line_width: float = 1.0,
) -> np.ndarray:
    """Rasterize a 2-D trajectory to a grayscale image.

    Parameters
    ----------
    xy         : Trajectory array of shape ``(T, 2)``.
    img_size   : Output image side length in pixels (height = width).
    line_width : Matplotlib line width for stroke rendering.

    Returns
    -------
    Float32 grayscale image of shape ``(img_size, img_size)`` with values in
    ``[0, 1]``.  Stroke pixels are near ``1``, background pixels near ``0``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy with shape (T, 2), got {xy.shape}")

    fig, ax = plt.subplots(
        figsize=(img_size / 100.0, img_size / 100.0), dpi=100
    )
    ax.plot(xy[:, 0], xy[:, 1], color="black", lw=float(line_width))

    margin = 0.05
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba())          # [H, W, 4] RGBA
        gray = buf[:, :, :3].mean(axis=2).astype(np.float32)
    else:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        argb = buf.reshape(h, w, 4)
        gray = argb[:, :, 1:].mean(axis=2).astype(np.float32)

    image = (255.0 - gray) / 255.0  # invert: stroke=1, background=0
    plt.close(fig)
    return image.astype(np.float32)
