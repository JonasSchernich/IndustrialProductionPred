# src/modeling/linear.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from .base import TSRegressor


class ElasticNetWrapper(TSRegressor):
    """
    ElasticNet mit optionaler Standardisierung (für varianz-/skalen-sensible Setups).
    """
    def __init__(self, alpha: float = 1e-3, l1_ratio: float = 0.5, max_iter: int = 5000, standardize: bool = False, random_state: int = 42):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.standardize = bool(standardize)
        self.random_state = int(random_state)
        self._pipe: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        steps = []
        if self.standardize:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("enet", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter, random_state=self.random_state)))
        self._pipe = Pipeline(steps=steps)
        self._pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipe.predict(X)


class PLSWrapper(TSRegressor):
    """
    PLSRegression als sklearn-kompatibler Regressor (optional ohne Scaling).
    Achtung: PLS ist skalen-sensibel; ggf. vorher StandardScaler in Feature-Pipeline nutzen.
    """
    def __init__(self, n_components: int = 2, scale: bool = False):
        self.n_components = int(n_components)
        self.scale = bool(scale)
        self._pls: Optional[PLSRegression] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        self._pls.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # sklearns PLSRegression.predict gibt 2D-Array; flache 1D zurückgeben
        preds = self._pls.predict(X)
        return preds.ravel()

# --- Backwards-compatibility aliases (expected by other modules)
ElasticNetRegressor = ElasticNetWrapper
PLSRegressor = PLSWrapper
