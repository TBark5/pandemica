"""Helpers shared by the analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.style import RESULTS_DIR, ensure_dirs


def _to_builtin(obj: Any) -> Any:
    """Convert numpy scalars/arrays so they can be written as JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(name: str, payload: dict) -> Path:
    """Write ``payload`` to ``results/<name>.json`` (pretty-printed)."""
    ensure_dirs()
    path = RESULTS_DIR / f"{name}.json"
    text = json.dumps(_to_builtin(payload), indent=2)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def save_csv(name: str, df: pd.DataFrame) -> Path:
    """Write ``df`` to ``results/<name>.csv``."""
    ensure_dirs()
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def load_json(name: str) -> dict:
    """Read ``results/<name>.json``."""
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
