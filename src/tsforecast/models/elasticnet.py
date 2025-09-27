# elasticnet.py
from __future__ import annotations
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def make_elasticnet(params: Dict[str, Any]):
    p = dict(params)
    standardize = bool(p.pop("standardize", True))

    # robuste Defaults (werden vom Grid überschrieben, wenn gesetzt)
    p.setdefault("max_iter", 20000)
    p.setdefault("tol", 1e-3)              # konservativ, konvergiert schneller
    p.setdefault("selection", "cyclic")    # "random" kann bei sehr großer Dim helfen
    p.setdefault("random_state", 42)       # wirkt nur bei selection="random"

    model = ElasticNet(**p)
    if standardize:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model
