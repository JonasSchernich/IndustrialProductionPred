from __future__ import annotations
from typing import Dict, Any, Callable

def _lower(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "")

def _make_elasticnet(params: Dict[str, Any]):
    from .elasticnet import make_elasticnet
    return make_elasticnet(params)

def _make_random_forest(params: Dict[str, Any]):
    try:
        from .random_forest import make_random_forest
    except Exception as e:
        raise ImportError("random_forest nicht verfügbar (scikit-learn?)") from e
    return make_random_forest(params)

def _make_xgboost(params: Dict[str, Any]):
    try:
        from .xgboost import make_xgboost
    except Exception as e:
        raise ImportError("xgboost nicht verfügbar (pip install xgboost)") from e
    return make_xgboost(params)

def _make_lightgbm(params: Dict[str, Any]):
    try:
        from .lightgbm import make_lightgbm
    except Exception as e:
        raise ImportError("lightgbm nicht verfügbar (pip install lightgbm)") from e
    return make_lightgbm(params)

def _make_mean(params: Dict[str, Any]):
    from .baselines import MeanModel
    return MeanModel(**params)

def _make_randomwalk(params: Dict[str, Any]):
    from .baselines import RandomWalkModel
    return RandomWalkModel(**params)

def _make_ar1(params: Dict[str, Any]):
    from .baselines import AR1Model
    return AR1Model(**params)

def _make_tabpfn(params: Dict[str, Any]):
    try:
        from .tabpfn import TabPFNEstimator
    except Exception as e:
        raise ImportError("TabPFN nicht verfügbar (pip install tabpfn)") from e
    return TabPFNEstimator(**params)

def _make_chronos(params: Dict[str, Any]):
    try:
        from .chronos import ChronosRegressor
    except Exception as e:
        raise ImportError("Chronos nicht verfügbar (pip install chronos-forecasting torch)") from e
    return ChronosRegressor(**params)

_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "elasticnet": _make_elasticnet,
    "randomforest": _make_random_forest,
    "rf": _make_random_forest,
    "xgboost": _make_xgboost,
    "xgb": _make_xgboost,
    "lightgbm": _make_lightgbm,
    "lgbm": _make_lightgbm,
    "mean": _make_mean,
    "avg": _make_mean,
    "average": _make_mean,
    "randomwalk": _make_randomwalk,
    "rw": _make_randomwalk,
    "ar1": _make_ar1,
    "tabpfn": _make_tabpfn,
    "chronos": _make_chronos,
    "naive": _make_randomwalk,
    "ar(1)": _make_ar1,
}

def build_estimator(model_name: str, params: Dict[str, Any]):
    key = _lower(model_name)
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unbekanntes Modell '{model_name}'. Verfügbar: {available}")
    return _REGISTRY[key](params or {})
