"""CASIA-OLHWDB1.1 dataset loading, collation, and PyTorch Dataset.

This module consolidates:
  - ``potreader.py``     : binary ``.pot`` file parser and normalization
  - ``casia_dataset.py`` : PyTorch Dataset wrapping variable-length trajectories
  - ``casia_collate.py`` : collate function for padding variable-length batches

Directory layout expected on disk::

    <raw_root>/
        OLHWDB1.1trn_pot/*.pot
        OLHWDB1.1tst_pot/*.pot
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# .pot binary reader (formerly potreader.py)
# ---------------------------------------------------------------------------

# Binary format of one CASIA online isolated character record:
#   sample_size  : 2 bytes, little-endian unsigned
#   tag_code     : 4 bytes  (character label, raw bytes)
#   stroke_number: 2 bytes, little-endian unsigned
#   coordinates  : (sample_size - 8) bytes, pairs of signed 16-bit ints (x, y)
# Stroke end marker : (x=-1, y=0)
# Character end marker: (x=-1, y=-1)

_POT_FIELD_BYTES = {
    "sample_size": 2,
    "tag_code": 4,
    "stroke_number": 2,
    "coordinate": 2,  # per component (x or y)
}


def read_pot(path: str | Path) -> Tuple[List, List, List, List]:
    """Read one ``.pot`` file and return its records.

    Returns
    -------
    sample_size_list    : list[int]
    tag_code_list       : list[bytes]
    stroke_number_list  : list[int]
    coords_list         : list[list[list[int, int]]]
    """
    sample_size_list: List[int] = []
    tag_code_list: List[bytes] = []
    stroke_number_list: List[int] = []
    coords_list: List[List] = []

    with open(path, "rb") as f:
        data = f.read()

    idx = 0
    while idx < len(data):
        # sample_size
        sample_size = int.from_bytes(data[idx : idx + 2], "little", signed=False)
        idx += 2
        sample_size_list.append(sample_size)

        # tag_code (raw label bytes)
        tag_code = bytes(data[idx : idx + 4])
        idx += 4
        tag_code_list.append(tag_code)

        # stroke_number
        stroke_number = int.from_bytes(data[idx : idx + 2], "little", signed=False)
        idx += 2
        stroke_number_list.append(stroke_number)

        # coordinates: (sample_size - 8) bytes → (sample_size-8)/4 points
        n_coord_bytes = sample_size - 8
        coords: List[List[int]] = []
        consumed = 0
        while consumed < n_coord_bytes:
            x = int.from_bytes(data[idx : idx + 2], "little", signed=True)
            idx += 2
            y = int.from_bytes(data[idx : idx + 2], "little", signed=True)
            idx += 2
            coords.append([x, y])
            consumed += 4
        coords_list.append(coords)

    return sample_size_list, tag_code_list, stroke_number_list, coords_list


def _remove_end_markers(coords_list: List[List]) -> List[List]:
    """Strip CASIA stroke/character end markers ``(-1,0)`` and ``(-1,-1)``."""
    markers = {(-1, 0), (-1, -1)}
    return [
        [xy for xy in sample if tuple(xy) not in markers]
        for sample in coords_list
    ]


def _normalize_sample(sample: List, eps: float = 1e-6) -> List:
    """Normalize one sample to ``[-1, +1]`` per axis independently.

    Constant axes are set to 0.
    """
    if not sample:
        return sample
    arr = np.asarray(sample, dtype=np.float32)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    spans = maxs - mins
    norm = np.empty_like(arr)
    stable = spans > eps
    norm[:, stable] = 2.0 * ((arr[:, stable] - mins[stable]) / spans[stable]) - 1.0
    norm[:, ~stable] = 0.0
    return norm.tolist()


def _normalize_dataset(coords_list: List[List], eps: float = 1e-6) -> List[List]:
    return [_normalize_sample(s, eps=eps) for s in coords_list]


def _read_pot_dir(pot_dir: Path) -> Tuple[List, List, List, List]:
    """Read all ``.pot`` files from a directory, sorted by filename."""
    if not pot_dir.exists():
        raise FileNotFoundError(f"CASIA pot directory not found: {pot_dir}")
    pot_files = sorted(pot_dir.glob("*.pot"))
    if not pot_files:
        raise FileNotFoundError(f"No .pot files found in: {pot_dir}")

    all_sizes: List[int] = []
    all_tags: List[bytes] = []
    all_strokes: List[int] = []
    all_coords: List[List] = []
    for p in pot_files:
        sizes, tags, strokes, coords = read_pot(p)
        all_sizes.extend(sizes)
        all_tags.extend(tags)
        all_strokes.extend(strokes)
        all_coords.extend(coords)
    return all_sizes, all_tags, all_strokes, all_coords


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def _label_to_token(label) -> str:
    """Convert a raw label to a stable JSON-serializable string token."""
    if isinstance(label, (bytes, bytearray)):
        return bytes(label).hex()
    return str(label)


def build_train_label_map(
    train_labels: List,
    save_path: Optional[str | Path] = None,
) -> Dict[str, int]:
    """Build a class-id mapping from train labels only.

    Parameters
    ----------
    train_labels : Iterable of raw label objects.
    save_path    : If given, write the map to a JSON file at this path.

    Returns
    -------
    dict mapping token string -> class integer id.
    """
    tokens = sorted({_label_to_token(lbl) for lbl in train_labels})
    label_to_id = {tok: idx for idx, tok in enumerate(tokens)}
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label_to_id": label_to_id,
            "id_to_label": tokens,
            "num_classes": len(tokens),
            "token_format": "hex_if_bytes_else_str",
        }
        save_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return label_to_id


def load_label_map(label_map_path: str | Path) -> Dict[str, int]:
    """Load a label-map JSON created by :func:`build_train_label_map`."""
    path = Path(label_map_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "label_to_id" not in payload:
        raise ValueError(f"Invalid label map (missing 'label_to_id'): {path}")
    return payload["label_to_id"]


def encode_labels_with_map(
    labels: List,
    label_to_id: Dict[str, int],
    split_name: str = "split",
    strict: bool = True,
) -> Tuple[np.ndarray, List]:
    """Encode raw labels to integer class ids using ``label_to_id``.

    Parameters
    ----------
    labels      : Iterable of raw label objects.
    label_to_id : Mapping from token string to integer class id.
    split_name  : Name of the split (used in error messages).
    strict      : If ``True``, raise on unknown labels; otherwise encode as -1.

    Returns
    -------
    encoded : int64 array of class ids.
    missing : list of (index, token) for labels not found in the map.
    """
    encoded: List[int] = []
    missing: List[Tuple[int, str]] = []
    for idx, lbl in enumerate(labels):
        token = _label_to_token(lbl)
        class_id = label_to_id.get(token)
        if class_id is None:
            missing.append((idx, token))
            if strict:
                raise ValueError(
                    f"Unseen label token in {split_name}: index={idx}, token={token}. "
                    "Use a compatible label map or disable strict mode."
                )
            encoded.append(-1)
        else:
            encoded.append(class_id)
    return np.asarray(encoded, dtype=np.int64), missing


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def load_casia(
    raw_root: Optional[str | Path] = None,
    train_dir: str = "OLHWDB1.1trn_pot",
    test_dir: str = "OLHWDB1.1tst_pot",
    normalize: bool = True,
) -> Tuple[List, List, List, List]:
    """Load CASIA OLHWDB1.1 raw data from pot directories.

    Parameters
    ----------
    raw_root  : Directory containing ``train_dir`` and ``test_dir``.
    train_dir : Subdirectory name for training pot files.
    test_dir  : Subdirectory name for test pot files.
    normalize : If ``True``, normalize each sample to ``[-1, 1]`` per axis.

    Returns
    -------
    train_data   : list of variable-length coordinate lists.
    train_labels : list of raw tag-code bytes (labels).
    test_data    : list of variable-length coordinate lists.
    test_labels  : list of raw tag-code bytes (labels).
    """
    raw_root = Path(raw_root) if raw_root is not None else _default_raw_root()
    _, train_labels, _, train_data = _read_pot_dir(raw_root / train_dir)
    _, test_labels, _, test_data = _read_pot_dir(raw_root / test_dir)

    train_data = _remove_end_markers(train_data)
    test_data = _remove_end_markers(test_data)
    if normalize:
        train_data = _normalize_dataset(train_data)
        test_data = _normalize_dataset(test_data)

    return train_data, train_labels, test_data, test_labels


def _default_raw_root() -> Path:
    # src/aite/data/casia.py -> CASIA/casia/raw relative to repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "CASIA" / "casia" / "raw"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CASIADataset(Dataset):
    """CASIA-OLHWDB1.1 isolated character dataset for variable-length ``(T, 2)`` trajectories.

    Each item is a dict::

        {
            "ink"    : FloatTensor[T, 2],
            "length" : int,
            "target" : int,
            "index"  : int,
        }
    """

    def __init__(
        self,
        split: str,
        data: List,
        targets: np.ndarray,
        drop_unknown: bool = False,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got: {split}")
        if len(data) != len(targets):
            raise ValueError(
                f"Data/targets length mismatch for split={split}: "
                f"{len(data)} != {len(targets)}"
            )

        self.split = split
        self._data = data
        self._targets = np.asarray(targets, dtype=np.int64)
        self._indices = np.arange(len(self._data), dtype=np.int64)

        if drop_unknown:
            self._indices = self._indices[self._targets >= 0]

        if len(self._indices) == 0:
            raise ValueError(f"No valid samples available in split={split}.")

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: idx={idx}, len={len(self)}")

        original_index = int(self._indices[idx])
        target = int(self._targets[original_index])
        if target < 0:
            raise ValueError(
                f"Unknown target class at split={self.split}, index={original_index}"
            )

        arr = np.asarray(self._data[original_index], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Bad trajectory shape at split={self.split}, index={original_index}: "
                f"{arr.shape} (expected (T, 2))"
            )
        if arr.shape[0] == 0:
            raise ValueError(
                f"Empty trajectory at split={self.split}, index={original_index}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"Non-finite value in trajectory at split={self.split}, "
                f"index={original_index}"
            )

        return {
            "ink": torch.from_numpy(arr),
            "length": int(arr.shape[0]),
            "target": target,
            "index": original_index,
        }


def build_casia_datasets(
    raw_root: str | Path,
    normalize: bool = True,
    label_map_path: Optional[str | Path] = None,
    save_label_map_path: Optional[str | Path] = None,
    strict_test_labels: bool = True,
) -> Tuple["CASIADataset", "CASIADataset", Dict]:
    """Load CASIA raw data and return train/test :class:`CASIADataset` objects.

    Parameters
    ----------
    raw_root            : Path to the directory containing the ``.pot`` subdirs.
    normalize           : Apply per-sample ``[-1, 1]`` normalization.
    label_map_path      : Pre-built label map JSON.  If ``None``, a new map is
                          built from training labels.
    save_label_map_path : Where to save the auto-built label map.
    strict_test_labels  : Raise on test labels unseen in the train map when
                          ``True``; encode as -1 when ``False``.

    Returns
    -------
    train_ds : :class:`CASIADataset` for the training split.
    test_ds  : :class:`CASIADataset` for the test split.
    metadata : dict with summary statistics.
    """
    raw_root_path = Path(raw_root)
    train_data, train_labels_raw, test_data, test_labels_raw = load_casia(
        raw_root=raw_root_path,
        normalize=normalize,
    )

    if label_map_path is not None:
        label_to_id = load_label_map(label_map_path)
        label_map_used = str(label_map_path)
    else:
        if save_label_map_path is None:
            # Default: alongside the raw root
            save_label_map_path = (
                raw_root_path.parents[1] / "audit_report" / "casia_label_map_train_only.json"
            )
        label_to_id = build_train_label_map(
            train_labels_raw, save_path=save_label_map_path
        )
        label_map_used = str(save_label_map_path)

    train_targets, train_missing = encode_labels_with_map(
        train_labels_raw, label_to_id, split_name="train", strict=True
    )
    test_targets, test_missing = encode_labels_with_map(
        test_labels_raw, label_to_id, split_name="test", strict=strict_test_labels
    )

    if train_missing:
        raise RuntimeError("Unexpected missing labels in train split.")

    train_ds = CASIADataset("train", train_data, train_targets, drop_unknown=False)
    test_ds = CASIADataset(
        "test", test_data, test_targets, drop_unknown=(not strict_test_labels)
    )

    metadata: Dict = {
        "num_train_samples": len(train_ds),
        "num_test_samples": len(test_ds),
        "num_classes": int(len(label_to_id)),
        "label_map_path": label_map_used,
        "num_unknown_test_labels": int(np.sum(test_targets < 0)),
    }
    return train_ds, test_ds, metadata


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def casia_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-length ``(T, 2)`` trajectories into padded batch tensors.

    Returns a dict with keys:

    * ``"ink"``     : ``FloatTensor[B, T_max, 2]``
    * ``"lengths"`` : ``LongTensor[B]``
    * ``"mask"``    : ``BoolTensor[B, T_max]`` — ``True`` for valid timesteps
    * ``"targets"`` : ``LongTensor[B]``
    * ``"indices"`` : ``LongTensor[B]``
    """
    if len(batch) == 0:
        raise ValueError("Empty batch received in casia_collate_fn.")

    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    targets = torch.tensor([x["target"] for x in batch], dtype=torch.long)
    indices = torch.tensor([x["index"] for x in batch], dtype=torch.long)

    bsz = len(batch)
    tmax = int(lengths.max().item())
    ink = torch.zeros((bsz, tmax, 2), dtype=torch.float32)
    mask = torch.zeros((bsz, tmax), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq: torch.Tensor = item["ink"]
        if seq.ndim != 2 or seq.shape[1] != 2:
            raise ValueError(
                f"Bad sequence shape in batch item {i}: {tuple(seq.shape)}"
            )
        t = int(seq.shape[0])
        ink[i, :t] = seq
        mask[i, :t] = True

    return {
        "ink": ink,
        "lengths": lengths,
        "mask": mask,
        "targets": targets,
        "indices": indices,
    }
