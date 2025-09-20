
from __future__ import annotations
from typing import Dict, Any
from .elasticnet import make_elasticnet
from .baselines import MeanModel, RandomWalkModel, AR1Model
from .random_forest import make_random_forest
from .xgboost import make_xgboost
from .lightgbm import make_lightgbm

def build_estimator(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "elasticnet":
        return make_elasticnet(params)
    if name in ("mean", "avg", "average"):
        return MeanModel()
    if name in ("randomwalk", "rw", "naive"):
        return RandomWalkModel()
    if name in ("ar1", "ar(1)"):
        fit_intercept = params.get("fit_intercept", True)
        return AR1Model(fit_intercept=fit_intercept)
    if name in ("rf", "randomforest", "random_forest"):
        return make_random_forest(params)
    if name in ("xgb", "xgboost"):
        return make_xgboost(params)
    if name in ("lgbm", "lightgbm"):
        return make_lightgbm(params)
    raise NotImplementedError(f"Model '{name}' not implemented.")
