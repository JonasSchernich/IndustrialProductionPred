
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
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("y is empty")
        self.last_ = float(y[-1])
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
        y = np.asarray(y, dtype=float)
        if y.size < 2:
            raise ValueError("Need at least 2 observations to fit AR(1).")
        y_lag = y[:-1]
        y_cur = y[1:]
        if self.fit_intercept:
            import numpy as np
            Xmat = np.column_stack([np.ones_like(y_lag), y_lag])
            beta, *_ = np.linalg.lstsq(Xmat, y_cur, rcond=None)
            self.intercept_, self.coef_ = float(beta[0]), float(beta[1])
        else:
            import numpy as np
            Xmat = y_lag.reshape(-1, 1)
            beta, *_ = np.linalg.lstsq(Xmat, y_cur, rcond=None)
            self.intercept_, self.coef_ = 0.0, float(beta[0])
        self.y_last_ = float(y[-1])
        return self
    def predict(self, X):
        n = getattr(X, 'shape', [len(X)])[0]
        yhat1 = self.intercept_ + (self.coef_ * self.y_last_)
        return np.full(n, yhat1, dtype=float)
