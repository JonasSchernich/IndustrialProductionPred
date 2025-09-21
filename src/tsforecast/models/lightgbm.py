
from __future__ import annotations
from typing import Dict, Any
try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None
    _LGBM_IMPORT_ERROR = e

def make_lightgbm(params: Dict[str, Any]):
    if LGBMRegressor is None:
        raise ImportError(f"lightgbm is not installed: {_LGBM_IMPORT_ERROR}")
    p = dict(params)
    p.setdefault("n_estimators", 500)
    p.setdefault("learning_rate", 0.05)
    p.setdefault("num_leaves", 15)
    p.setdefault("min_data_in_leaf", 5)
    p.setdefault("subsample", 0.8)
    p.setdefault("colsample_bytree", 0.8)
    p.setdefault("objective", "regression")
    return LGBMRegressor(**p)
