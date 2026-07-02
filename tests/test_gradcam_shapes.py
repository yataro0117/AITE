"""Tests for Grad-CAM output shapes and normalization."""

import numpy as np

from aite.models.casia import get_casia_model
from aite.models.unipen import get_unipen_model
from aite.salience.gradcam import compute_gradcam, find_last_conv1d


def _unipen_input(T: int = 50) -> np.ndarray:
    return np.random.uniform(-1, 1, (T, 2)).astype(np.float32)


def test_find_last_conv1d_returns_string() -> None:
    """find_last_conv1d should return a non-empty string for a CNN model."""
    model = get_unipen_model("cnn3", dataset="1a")
    name = find_last_conv1d(model)
    assert isinstance(name, str) and len(name) > 0


def test_gradcam_output_shape_unipen() -> None:
    """compute_gradcam on a Unipen CNN3 model should return cam[T] and logits[K]."""
    model = get_unipen_model("cnn3", dataset="1a")
    x = _unipen_input(50)
    cam, logits = compute_gradcam(model, x)
    assert cam.shape == (50,)
    assert logits.shape == (10,)


def test_gradcam_cam_normalized() -> None:
    """Returned CAM values should be in [-1, 1]."""
    model = get_unipen_model("cnn3", dataset="1a")
    cam, _ = compute_gradcam(model, _unipen_input(50))
    assert cam.min() >= -1.0 - 1e-5
    assert cam.max() <= 1.0 + 1e-5


def test_gradcam_logits_shape() -> None:
    """Returned logits should have shape [num_classes]."""
    model = get_unipen_model("cnn4", dataset="1b")
    _, logits = compute_gradcam(model, _unipen_input(50))
    assert logits.shape == (26,)


def test_gradcam_casia_model() -> None:
    """compute_gradcam with casia_model=True should handle variable-length input."""
    model = get_casia_model("cnn1d", num_classes=100)
    T = 64
    x = np.random.uniform(-1, 1, (T, 2)).astype(np.float32)
    cam, logits = compute_gradcam(model, x, casia_model=True)
    assert cam.shape == (T,)
    assert logits.shape == (100,)


def test_gradcam_relu_nonnegative() -> None:
    """With relu=True, all CAM values should be >= 0."""
    model = get_unipen_model("cnn3", dataset="1a")
    cam, _ = compute_gradcam(model, _unipen_input(50), relu=True)
    assert cam.min() >= 0.0


def test_gradcam_explicit_class_idx() -> None:
    """Passing an explicit class_idx should not raise."""
    model = get_unipen_model("cnn3", dataset="1a")
    cam, logits = compute_gradcam(model, _unipen_input(50), class_idx=3)
    assert cam.shape == (50,)
    assert logits.shape == (10,)
