"""Experiment metadata serialization utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def save_metadata(
    path: str | Path,
    config: Dict[str, Any],
    git_hash: Optional[str] = None,
) -> None:
    """Save experiment configuration and optional git commit hash as JSON.

    Parameters
    ----------
    path     : Destination file path (created including parent directories).
    config   : Arbitrary dict of experiment hyperparameters/settings.
    git_hash : Git commit hash to record.  If ``None``, attempts to read it
               from the current working directory via ``git rev-parse HEAD``.
               If that fails (e.g. not a git repo), the field is omitted.

    Notes
    -----
    Values in ``config`` must be JSON-serializable.  Non-serializable objects
    (e.g. ``Path``) are converted to strings via a custom encoder.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if git_hash is None:
        git_hash = _try_get_git_hash()

    payload: Dict[str, Any] = {}
    if git_hash is not None:
        payload["git_hash"] = git_hash
    payload.update(config)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_get_git_hash() -> Optional[str]:
    """Return the current git HEAD commit hash, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder: convert Path and other objects to strings."""
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
