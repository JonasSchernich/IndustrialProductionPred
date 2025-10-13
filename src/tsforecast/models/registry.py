from __future__ import annotations
from typing import Dict, Any
import numpy as np

def build_estimator(model_name: str, params: Dict[str, Any]):
    name = (model_name or "").lower()

    if name in ("xgb", "xgboost"):
        from .xgboost import build_estimator as _bx
        return _bx(params)

    if name in ("lgbm", "lightgbm"):
        from .lightgbm import build_estimator as _bl
        return _bl(params)

    if name in ("elasticnet", "en", "enet"):
        from .elasticnet import build_estimator as _be
        return _be(params)

    if name in ("tabpfn", "tab-pfn"):
        from .tabpfn import build_estimator as _bt
        return _bt(params)

    if name in ("mean", "avg", "average"):
        from .baselines import MeanModel
        return MeanModel()
    if name in ("randomwalk", "rw", "naive"):
        from .baselines import RandomWalkModel
        return RandomWalkModel()
    if name in ("ar1", "ar(1)"):
        from .baselines import AR1Model
        return AR1Model()
    if name in ("pls_en", "pls+en", "en_pls", "plsen"):
        from .pls_en import build_estimator as _bp
        return _bp(params)

    raise ValueError(f"unknown model_name='{model_name}'")

def supports_es(model_name: str) -> bool:
    name = (model_name or "").lower()
    return name in {"xgb", "xgboost", "lgbm", "lightgbm"}
