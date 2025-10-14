# pls_en.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet

class PLSEN(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=4, alpha=1.0, l1_ratio=0.5,
                 max_iter=20000, tol=1e-3, random_state=42):
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)  # train-only
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(Xs, y)
        T = self.pls_.transform(Xs)
        self.en_ = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio,
            max_iter=self.max_iter, tol=self.tol, selection="cyclic",
            random_state=self.random_state
        )
        self.en_.fit(T, y)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        T = self.pls_.transform(Xs)
        return self.en_.predict(T)

def build_estimator(params: Dict[str, Any]):
    p = dict(params or {})
    p.setdefault("n_components", 4)
    p.setdefault("alpha", 1.0)
    p.setdefault("l1_ratio", 0.5)
    p.setdefault("max_iter", 20000)
    p.setdefault("tol", 1e-3)
    p.setdefault("random_state", 42)
    return PLSEN(**p)
