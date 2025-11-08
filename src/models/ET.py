# src/models/ET.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


class ForecastModel:
    """
    ExtraTreesRegressor for the tuning pipeline.
    - .fit(X, y, sample_weight=None)
    - .predict_one(x_row)
    - .get_feature_importances(feature_names)
    - HPs: n_estimators, max_features, min_samples_leaf, min_samples_split, max_depth, seed
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self._reg = None
        self._backend_name = "extra_trees"
        self._feature_names: Optional[List[str]] = None
        self._importances: Optional[np.ndarray] = None

        # Extract HPs
        self._n_estimators = int(self.params.get('n_estimators', 100))
        self._max_features = self.params.get('max_features', 'sqrt')  # Default from sklearn/thesis
        self._min_samples_leaf = int(self.params.get('min_samples_leaf', 1))
        self._min_samples_split = int(self.params.get('min_samples_split', 2))
        self._max_depth = self.params.get('max_depth')  # None by default
        if self._max_depth is not None:
            self._max_depth = int(self._max_depth)

        # KORRIGIERT: Seed aus params (Fallback 42)
        self._seed = int(self.params.get('seed', 42))

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=float)
        # nan_to_num hier als letzte Sicherheitsmaßnahme vor dem Fitten
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def fit(self, X, y, sample_weight=None):

        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_np = X.values
        elif hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self._feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

        X_np = self._clean(X_np)
        y_np = np.asarray(y, dtype=float).ravel()

        self._reg = ExtraTreesRegressor(
            n_estimators=self._n_estimators,
            max_features=self._max_features,
            min_samples_leaf=self._min_samples_leaf,
            min_samples_split=self._min_samples_split,
            max_depth=self._max_depth,
            random_state=self._seed,  # <-- KORRIGIERT
            n_jobs=1  # Safer for nested parallelism
        )

        self._reg.fit(X_np, y_np, sample_weight=sample_weight)

        # Store feature importances
        self._importances = self._reg.feature_importances_

        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")

        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_np = self._clean(X_np)
        return self._reg.predict(X_np)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances as a dictionary."""
        if self._importances is None:
            # raise RuntimeError("Model not fitted or importances not available.")
            return {}  # Leeres Dict zurückgeben, wenn nicht verfügbar

        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances))]
        return dict(zip(names, self._importances))