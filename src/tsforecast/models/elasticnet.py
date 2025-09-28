# elasticnet.py
from __future__ import annotations
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def make_elasticnet(params: Dict[str, Any]):
    p = dict(params or {})
    standardize = bool(p.pop("standardize", True))

    # robuste Defaults (werden vom Grid überschrieben)
    p.setdefault("max_iter", 20000)
    p.setdefault("tol", 1e-3)
    p.setdefault("selection", "cyclic")
    p.setdefault("random_state", 42)

    model = ElasticNet(**p)
    if standardize:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model

def build_estimator(params: Dict[str, Any]):
    """Registry-Factory Hook."""
    return make_elasticnet(params or {})
