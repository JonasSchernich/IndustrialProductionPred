# models/tabpfn.py
from typing import Optional
import numpy as np

class TabPFNEstimator:
    def __init__(self, **params):
        try:
            from tabpfn import TabPFNRegressor
        except Exception as e:
            raise ImportError("pip install tabpfn") from e
        self._params = params
        self._reg = None

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        p = dict(self._params)
        p.setdefault("device", "cpu")
        p.setdefault("ignore_pretraining_limits", True)
        self._reg = TabPFNRegressor(**p)
        self._reg.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self._reg.predict(np.asarray(X))
