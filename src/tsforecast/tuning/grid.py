from __future__ import annotations
from typing import Dict, Any, Iterable, List, Tuple
import numpy as np
import itertools

def _to_iter(v: Any) -> Iterable:
    if isinstance(v, (list, tuple)):
        return v
    return [v]

def expand_grid(grid: Any) -> List[Dict[str, Any]]:
    """Dict[param]->values -> List[dict]; List[dict] passt unverändert durch."""
    if isinstance(grid, list):
        return list(grid)
    if not isinstance(grid, dict):
        raise TypeError("grid must be dict or list of dicts")
    keys = list(grid.keys())
    vals = [list(_to_iter(grid[k])) for k in keys]
    combos = itertools.product(*vals)
    out: List[Dict[str, Any]] = []
    for tup in combos:
        out.append({k: v for k, v in zip(keys, tup)})
    return out

def sample_grid(grid: Any, n: int, seed: int | None = None) -> List[Dict[str, Any]]:
    """Zufällige Teilmenge aus expand_grid(grid), ohne Wiederholung."""
    hp_list = expand_grid(grid)
    total = len(hp_list)
    if n >= total:
        return hp_list
    rng = np.random.RandomState(None if seed is None else int(seed))
    idx = rng.choice(total, size=int(n), replace=False)
    return [hp_list[i] for i in idx]

def cartesian_size(grid: Dict[str, Any]) -> int:
    """Schnelle Größenabschätzung des Produkts (für Logging)."""
    if isinstance(grid, list):
        return len(grid)
    if not isinstance(grid, dict):
        return 0
    sz = 1
    for v in grid.values():
        sz *= len(list(_to_iter(v)))
    return int(sz)
