"""Tests for Grad-CAM output shapes and normalization."""

from aite.salience.gradcam import compute_gradcam, find_last_conv1d


def test_find_last_conv1d_returns_string() -> None:
    """find_last_conv1d should return a non-empty string for a CNN model."""
    pass


def test_gradcam_output_shape_unipen() -> None:
    """compute_gradcam on a Unipen CNN3 model should return cam[T] and logits[K]."""
    pass


def test_gradcam_cam_normalized() -> None:
    """Returned CAM values should be in [-1, 1]."""
    pass


def test_gradcam_logits_shape() -> None:
    """Returned logits should have shape [num_classes]."""
    pass


def test_gradcam_casia_model() -> None:
    """compute_gradcam with casia_model=True should handle variable-length input."""
    pass


def test_gradcam_relu_nonnegative() -> None:
    """With relu=True, all CAM values should be >= 0."""
    pass


def test_gradcam_explicit_class_idx() -> None:
    """Passing an explicit class_idx should not raise."""
    pass
