
from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MeanModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.mean_: Optional[float] = None
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("y is empty")
        self.mean_ = float(np.nanmean(y))
        return self
    def predict(self, X):
        n = getattr(X, 'shape', [len(X)])[0]
        return np.full(n, self.mean_, dtype=float)

class RandomWalkModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.last_: Optional[float] = None
    # baselines.py
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        valid = y[np.isfinite(y)]
        if valid.size == 0:
            raise ValueError("y has no finite values")
        self.last_ = float(valid[-1])
        return self

    def predict(self, X):
        n = getattr(X, 'shape', [len(X)])[0]
        return np.full(n, self.last_, dtype=float)

class AR1Model(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[float] = None
        self.intercept_: float = 0.0
        self.y_last_: Optional[float] = None
    def fit(self, X, y):
        # in baselines.AR1Model.fit(self, X, y)
        y = np.asarray(y, dtype=float)
        # gültige Nachbarpaare
        y0, y1 = y[:-1], y[1:]
        mask = np.isfinite(y0) & np.isfinite(y1)
        y0, y1 = y0[mask], y1[mask]
        if y1.size == 0:
            raise ValueError("Need ≥1 valid (y[t-1], y[t]) pair for AR(1).")
        if self.fit_intercept:
            Xmat = np.column_stack([np.ones_like(y0), y0])
            beta, *_ = np.linalg.lstsq(Xmat, y1, rcond=None)
            self.intercept_, self.coef_ = float(beta[0]), float(beta[1])
        else:
            Xmat = y0.reshape(-1, 1)
            beta, *_ = np.linalg.lstsq(Xmat, y1, rcond=None)
            self.intercept_, self.coef_ = 0.0, float(beta[0])
        # letzte gültige y
        valid = np.asarray(y[np.isfinite(y)], dtype=float)
        self.y_last_ = float(valid[-1])
        return self

    def predict(self, X):
        n = getattr(X, 'shape', [len(X)])[0]
        yhat1 = self.intercept_ + (self.coef_ * self.y_last_)
        return np.full(n, yhat1, dtype=float)
