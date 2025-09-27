# xgboost.py
from __future__ import annotations
from typing import Dict, Any
try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None
    _XGB_IMPORT_ERROR = e

def make_xgboost(params: Dict[str, Any]):
    if XGBRegressor is None:
        raise ImportError(f"xgboost is not installed: {_XGB_IMPORT_ERROR}")
    p = dict(params)
    use_gpu = bool(p.pop("use_gpu", False))

    # solide Defaults
    p.setdefault("n_estimators", 800)
    p.setdefault("learning_rate", 0.05)
    p.setdefault("max_depth", 4)
    p.setdefault("subsample", 0.8)
    p.setdefault("colsample_bytree", 0.8)
    p.setdefault("reg_lambda", 1.0)
    p.setdefault("reg_alpha", 0.0)
    p.setdefault("objective", "reg:squarederror")
    p.setdefault("random_state", 42)
    p.setdefault("n_jobs", -1)
    p.setdefault("verbosity", 0)

    if use_gpu:
        p["tree_method"] = "gpu_hist"
        p["predictor"] = "gpu_predictor"
        # optional: p["max_bin"] = 256
    else:
        p.setdefault("tree_method", "hist")

    return XGBRegressor(**p)
