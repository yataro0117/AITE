"""Salience (saliency map) methods for 1-D Conv trajectory models."""

from .gradcam import compute_gradcam, find_last_conv1d
from .input_gradient import compute_input_gradient

__all__ = [
    "compute_gradcam",
    "find_last_conv1d",
    "compute_input_gradient",
]
