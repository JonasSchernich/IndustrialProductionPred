# lightgbm.py
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
    use_gpu = bool(p.pop("use_gpu", False))

    # sinnvolle Defaults
    p.setdefault("n_estimators", 800)
    p.setdefault("learning_rate", 0.05)
    p.setdefault("num_leaves", 31)
    p.setdefault("min_data_in_leaf", 20)
    p.setdefault("subsample", 0.8)           # bagging_fraction
    p.setdefault("colsample_bytree", 0.8)    # feature_fraction
    p.setdefault("lambda_l2", 1.0)
    p.setdefault("objective", "regression")
    p.setdefault("random_state", 42)
    p.setdefault("n_jobs", -1)
    p.setdefault("verbosity", -1)
    # optional stabilisierend:
    p.setdefault("bagging_freq", 1)
    # p.setdefault("min_gain_to_split", 0.0) # default ok

    if use_gpu:
        p["device"] = "gpu"   # GPU-Build nötig

    return LGBMRegressor(**p)
