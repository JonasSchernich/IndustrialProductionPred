
from __future__ import annotations
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

def make_elasticnet(params: Dict[str, Any]):
    p = dict(params)
    standardize = bool(p.pop("standardize", True))
    model = ElasticNet(**p)
    if standardize:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model
