"""Tests for reproducibility utilities (seed setting and metadata saving)."""

import json

import numpy as np

from aite.reproducibility.metadata import save_metadata
from aite.reproducibility.seed import set_seed


def test_set_seed_runs_without_error() -> None:
    """set_seed should complete without raising for a valid seed value."""
    set_seed(42)


def test_set_seed_numpy_reproducible() -> None:
    """NumPy random draws should be identical after the same seed is set twice."""
    set_seed(123)
    a = np.random.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_set_seed_torch_reproducible() -> None:
    """PyTorch random draws should be identical after the same seed is set twice."""
    import torch

    set_seed(7)
    a = torch.rand(10)
    set_seed(7)
    b = torch.rand(10)
    assert torch.equal(a, b)


def test_save_metadata_creates_file(tmp_path) -> None:
    """save_metadata should create a JSON file at the given path."""
    path = tmp_path / "meta.json"
    save_metadata(path, {"lr": 1e-3}, git_hash="abc123")
    assert path.exists()


def test_save_metadata_content(tmp_path) -> None:
    """The saved JSON should contain the provided config keys."""
    path = tmp_path / "meta.json"
    save_metadata(path, {"lr": 1e-3, "arch": "cnn3"}, git_hash="abc123")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["lr"] == 1e-3
    assert data["arch"] == "cnn3"
    assert data["git_hash"] == "abc123"


def test_save_metadata_git_hash_optional(tmp_path) -> None:
    """save_metadata should work when git_hash is explicitly set to None."""
    path = tmp_path / "meta.json"
    save_metadata(path, {"seed": 42}, git_hash=None)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seed"] == 42


def test_save_metadata_creates_parent_dirs(tmp_path) -> None:
    """save_metadata should create parent directories if they do not exist."""
    path = tmp_path / "nested" / "sub" / "meta.json"
    save_metadata(path, {"k": "v"}, git_hash="abc123")
    assert path.exists()


def test_save_metadata_non_serializable_config(tmp_path) -> None:
    """Non-JSON-native values (e.g. Path) should be coerced via the custom encoder."""
    from pathlib import Path

    path = tmp_path / "meta.json"
    save_metadata(path, {"out_dir": Path("/tmp/x")}, git_hash="abc123")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["out_dir"] == "/tmp/x"
