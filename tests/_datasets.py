# -*- coding: utf-8 -*-
"""Shared dataset helpers for tests."""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import pytest

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_DATASET_SEARCH_PATHS: Iterable[Path] = (
    _REPO_ROOT / "datasets" / "experiments",
    _THIS_DIR / "datasets" / "experiments",
)
_PENGUIN_COLUMNS = [
    "X-Acc",
    "Y-Acc",
    "Z-Acc",
    "4",
    "5",
    "6",
    "7",
    "Pressure",
    "9",
]

def _find_in_search_paths(filename: str) -> Optional[Path]:
    for base in _DATASET_SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def require_dataset(filename: str) -> Path:
    """Return the path to `filename` within the experiments datasets, skipping if missing."""
    dataset_path = _find_in_search_paths(filename)
    if dataset_path is None:
        pytest.skip(f"Dataset '{filename}' not available", allow_module_level=True)
    return dataset_path


def read_penguin() -> Tuple[str, pd.DataFrame]:
    """Load the penguin dataset used across performance tests."""
    path = require_dataset("penguin.txt")
    series = pd.read_csv(
        path,
        names=_PENGUIN_COLUMNS,
        delimiter="\t",
        header=None,
    )
    ds_name = "Penguins (Longer Snippet)"
    return ds_name, series
