# src/modeling/base.py
from __future__ import annotations
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


def _to_series(x, name: str = "y") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x), name=name)


class TSRegressor(BaseEstimator, RegressorMixin):
    """
    Basisklasse für Zeitreihenregressoren (sklearn-kompatibel).
    Implementiert nur Save/Load-Helfer; fit/predict werden in Subklassen überschrieben.
    """

    # --- sklearn API ---
    def fit(self, X: pd.DataFrame, y: pd.Series):  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # --- Convenience ---
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TSRegressor":
        return joblib.load(path)

    # optional: ensure numpy arrays out
    @staticmethod
    def as_2d(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
