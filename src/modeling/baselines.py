# src/modeling/baselines.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import TSRegressor

class MeanForecast(TSRegressor):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mu_ = float(pd.Series(y).mean())
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu_, dtype=float)

class RandomWalk(TSRegressor):
    """y_{t+1} = y_t"""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.last_ = float(pd.Series(y).iloc[-1])
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.last_, dtype=float)

class AR1_OLS(TSRegressor):
    """Einfaches AR(1) über OLS: y_t = c + φ y_{t-1} + ε_t"""
    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(y).dropna()
        if len(y) < 3:
            self.c_, self.phi_ = 0.0, 0.0
            return self
        y_lag = y.shift(1).dropna()
        y_cur = y.reindex(y_lag.index)
        # Regress y_cur ~ [1, y_lag]
        A = np.column_stack([np.ones(len(y_lag)), y_lag.values])
        theta, *_ = np.linalg.lstsq(A, y_cur.values, rcond=None)
        self.c_, self.phi_ = float(theta[0]), float(theta[1])
        self.last_ = float(y.iloc[-1])
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Ein-Schritt-ahead Prognose ab letztem y
        yhat1 = self.c_ + self.phi_ * self.last_
        return np.full(len(X), float(yhat1), dtype=float)
