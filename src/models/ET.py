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
    - HPs: n_estimators, max_features, min_samples_leaf, min_samples_split, max_depth
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self._reg = None
        self._backend_name = "extra_trees"
        self._feature_names = None
        self._importances = None

        # Extract HPs
        self._n_estimators = int(self.params.get('n_estimators', 100))
        self._max_features = self.params.get('max_features', 'sqrt')  # Default from sklearn/thesis
        self._min_samples_leaf = int(self.params.get('min_samples_leaf', 1))
        self._min_samples_split = int(self.params.get('min_samples_split', 2))
        self._max_depth = self.params.get('max_depth')  # None by default
        if self._max_depth is not None:
            self._max_depth = int(self._max_depth)

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def fit(self, X, y, sample_weight=None):
        X = self._clean(X)
        y = np.asarray(y, dtype=float).ravel()

        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        elif hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
            X = X.values
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self._reg = ExtraTreesRegressor(
            n_estimators=self._n_estimators,
            max_features=self._max_features,
            min_samples_leaf=self._min_samples_leaf,
            min_samples_split=self._min_samples_split,
            max_depth=self._max_depth,
            random_state=42,
            n_jobs=1  # Safer for nested parallelism
        )

        self._reg.fit(X, y, sample_weight=sample_weight)

        # Store feature importances
        self._importances = dict(zip(self._feature_names, self._reg.feature_importances_))

        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")
        X = self._clean(X)
        return self._reg.predict(X)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances as a dictionary."""
        if self._importances is None:
            raise RuntimeError("Model not fitted or importances not available.")
        # Return copy
        return self._importances.copy()