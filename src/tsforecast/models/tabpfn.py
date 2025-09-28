from __future__ import annotations
from typing import Dict, Any
import numpy as np

class TabPFNEstimator:
    def __init__(self, **params):
        try:
            from tabpfn import TabPFNRegressor  # noqa: F401
        except Exception as e:
            raise ImportError("pip install tabpfn") from e
        self._params = dict(params)
        self._reg = None

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        import torch
        p = dict(self._params)
        use_gpu = bool(p.pop("use_gpu", False))
        if use_gpu and torch.cuda.is_available():
            p.setdefault("device", "cuda")
        else:
            p.setdefault("device", "cpu")
        p.setdefault("ignore_pretraining_limits", True)
        self._reg = TabPFNRegressor(**p)
        self._reg.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self._reg.predict(np.asarray(X))

def build_estimator(params: Dict[str, Any]):
    """Registry-Factory Hook."""
    return TabPFNEstimator(**(params or {}))
