from __future__ import annotations
from typing import Dict, Any
import warnings

def _default_params() -> Dict[str, Any]:
    return dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        tree_method="hist",
        predictor="auto",
    )

def build_estimator(params: Dict[str, Any]):
    try:
        import xgboost as xgb
    except Exception as e:
        raise ImportError("xgboost ist nicht installiert") from e

    p = _default_params()
    p.update({k: v for k, v in (params or {}).items() if v is not None})

    # GPU toggle (use_gpu=True → gpu_hist)
    use_gpu = bool(p.pop("use_gpu", False))
    if use_gpu:
        p["tree_method"] = "gpu_hist"
        p["predictor"] = "gpu_predictor"

    # XGBRegressor akzeptiert eval_set/early_stopping_rounds direkt
    est = xgb.XGBRegressor(**p)
    return est
