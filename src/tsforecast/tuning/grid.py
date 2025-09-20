
from itertools import product
from typing import Dict, Iterable

def expand_grid(grid: Dict[str, Iterable]):
    keys = list(grid.keys())
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
    for combo in product(*vals):
        yield dict(zip(keys, combo))
