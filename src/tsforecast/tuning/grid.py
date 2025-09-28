from __future__ import annotations
from typing import Dict, Any, Iterator, List
import itertools
import numpy as np

def _as_list(v: Any) -> List[Any]:
    if v is None:
        return [None]
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return list(v)
    return [v]

def expand_grid(grid: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Kartesisches Produkt über ein Hyperparameter-Grid.
    Werte dürfen Skalar oder Iterable sein. None bleibt als Wert erhalten.
    """
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    lists = [_as_list(grid[k]) for k in keys]
    for combo in itertools.product(*lists):
        yield {k: v for k, v in zip(keys, combo)}

def random_from_grid(grid: Dict[str, Any], n: int, seed: int | None = None) -> List[Dict[str, Any]]:
    """
    Schnelles Random-Sampling pro Schlüssel unabhängig (keine Kartesische Explosion nötig).
    Für große Grids praktisch als Vorschau; Duplikate möglich.
    """
    rng = np.random.RandomState(seed)
    keys = list(grid.keys())
    lists = [_as_list(grid[k]) for k in keys]
    out: List[Dict[str, Any]] = []
    for _ in range(int(n)):
        out.append({k: rng.choice(vals) for k, vals in zip(keys, lists)})
    return out
