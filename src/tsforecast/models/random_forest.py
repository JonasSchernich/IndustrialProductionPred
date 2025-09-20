
from __future__ import annotations
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor

def make_random_forest(params: Dict[str, Any]):
    p = dict(params)
    return RandomForestRegressor(**p)
