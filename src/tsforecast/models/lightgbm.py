from __future__ import annotations
from typing import Dict, Any

def _default_params() -> Dict[str, Any]:
    return dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        # device/device_type werden unten gemappt
    )

def build_estimator(params: Dict[str, Any]):
    try:
        import lightgbm as lgb
    except Exception as e:
        raise ImportError("lightgbm ist nicht installiert") from e

    p = _default_params()
    p.update({k: v for k, v in (params or {}).items() if v is not None})

    # GPU mapping (wir unterstützen beides: device='gpu' und device_type)
    device = str(p.pop("device", "")).lower()
    device_type = str(p.pop("device_type", "")).lower()
    if device == "gpu" or device_type == "gpu" or p.get("use_gpu", False):
        p["device_type"] = "gpu"  # LightGBM akzeptiert 'device_type' (>=v3), ältere akzeptieren 'device'
        p.setdefault("device", "gpu")  # harmless fallback

    # LGBMRegressor akzeptiert eval_set/early_stopping_rounds
    est = lgb.LGBMRegressor(**p)
    return est
