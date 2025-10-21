# src/models/LGBM.py
from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class ForecastModel:
    """
    LightGBM-Regressor for the tuning pipeline.
    - Handles early stopping internally using a validation tail.
    - .fit(X, y, sample_weight=None)
    - .predict_one(x_row)
    - .get_feature_importances(feature_names)
    - HPs: learning_rate, num_leaves, max_depth, subsample, colsample_bytree,
           min_child_samples, reg_alpha, reg_lambda, n_estimators,
           early_stopping_rounds, es_val_tail_size
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if lgb is None:
            raise ImportError("LightGBM not installed. Please `pip install lightgbm`.")

        self.params = dict(params or {})
        self._reg = None
        self._backend_name = "lightgbm"
        self._feature_names = None
        self._importances = None

        # Extract specific HPs for clarity
        self._learning_rate = float(self.params.get('learning_rate', 0.1))
        self._num_leaves = int(self.params.get('num_leaves', 31))
        self._max_depth = int(self.params.get('max_depth', -1))
        self._subsample = float(self.params.get('subsample', 0.8))
        self._colsample_bytree = float(self.params.get('colsample_bytree', 0.8))
        self._min_child_samples = int(self.params.get('min_child_samples', 20))
        self._reg_alpha = float(self.params.get('reg_alpha', 0.0))
        self._reg_lambda = float(self.params.get('reg_lambda', 0.0))
        self._n_estimators = int(self.params.get('n_estimators', 1000))
        self._early_stopping_rounds = int(self.params.get('early_stopping_rounds', 50))
        self._es_val_tail_size = int(self.params.get('es_val_tail_size', 24))  # Months for validation tail

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def fit(self, X, y, sample_weight=None):
        X = self._clean(X)
        y = np.asarray(y, dtype=float).ravel()
        n_samples = len(y)

        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        elif hasattr(X, 'columns'):  # Handles numpy structured arrays etc.
            self._feature_names = list(X.columns)
            X = X.values
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Prepare data and validation split for early stopping
        # Follows the logic from Methods
        val_size = min(self._es_val_tail_size, max(1, n_samples // 5))  # Ensure val_size is reasonable
        if n_samples <= val_size * 2:  # Not enough data for a meaningful split
            X_train, y_train, w_train = X, y, sample_weight
            X_val, y_val, w_val = None, None, None
            eval_set = None
            fit_params = {}
        else:
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]

            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight).ravel()
                w_train, w_val = sample_weight[:-val_size], sample_weight[-val_size:]
            else:
                w_train, w_val = None, None

            eval_set = [(X_val, y_val)]
            fit_params = {
                'eval_set': eval_set,
                'eval_sample_weight': [w_val] if w_val is not None else None,
                'callbacks': [
                    lgb.early_stopping(self._early_stopping_rounds, verbose=False)
                ]
            }

        self._reg = lgb.LGBMRegressor(
            learning_rate=self._learning_rate,
            num_leaves=self._num_leaves,
            max_depth=self._max_depth,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            min_child_samples=self._min_child_samples,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            n_estimators=self._n_estimators,
            random_state=42,
            n_jobs=1  # Safer for nested parallelism
        )

        self._reg.fit(X_train, y_train, sample_weight=w_train, **fit_params)

        # Store feature importances
        self._importances = dict(zip(self._feature_names, self._reg.feature_importances_))

        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")
        X = self._clean(X)
        # Use best_iteration_ if early stopping was used
        best_iter = getattr(self._reg, 'best_iteration_', None)
        return self._reg.predict(X, num_iteration=best_iter)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances as a dictionary."""
        if self._importances is None:
            raise RuntimeError("Model not fitted or importances not available.")
        # Return copy to prevent external modification
        return self._importances.copy()