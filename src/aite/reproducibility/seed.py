"""Global random seed initialization for reproducible experiments."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + all GPUs).

    Parameters
    ----------
    seed : Integer seed value.  Must be in ``[0, 2**32 - 1]``.

    Notes
    -----
    * Sets ``torch.backends.cudnn.deterministic = True`` and
      ``torch.backends.cudnn.benchmark = False`` to maximize reproducibility
      at the cost of some training speed.
    * Does **not** control ``os.environ["PYTHONHASHSEED"]`` because that must
      be set before the Python interpreter starts.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
