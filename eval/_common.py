"""Small helpers shared by the eval scripts (numpy only)."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Sequence

import numpy as np


def trailing_int(name: str) -> int:
    """Trailing integer of a seed directory name (``seed-2`` -> 2, ``2`` -> 2).

    Names without a trailing integer sort first (-1), then alphabetically via the
    caller's stable sort.
    """
    match = re.search(r'(\d+)$', name)
    return int(match.group(1)) if match else -1


def sort_seeds(names: Sequence[str]) -> List[str]:
    """Sort seed names by trailing integer (as the original scripts did)."""
    return sorted(sorted(names), key=trailing_int)


def list_subdirs(path: str) -> List[str]:
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def mean_std(values: Sequence[float]) -> Dict[str, object]:
    """Mean and *population* std (np.std, ddof=0) -- what the paper scripts used."""
    arr = np.asarray(list(values), dtype=float)
    return {'mean': float(arr.mean()), 'std': float(arr.std()), 'n': int(arr.size),
            'values': [float(v) for v in arr]}


def save_json(obj, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
