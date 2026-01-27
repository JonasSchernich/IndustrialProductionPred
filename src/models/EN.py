from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


class ForecastModel:
    """
    ElasticNet wrapper with an ET-like interface
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._reg: Optional[ElasticNet] = None
        self._backend_name: str = "elastic_net"
        self._feature_names: Optional[List[str]] = None
        self._importances: Optional[np.ndarray] = None

        # Scaling params (train-only z-standardization)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None

        # --- Map thesis params -> sklearn params ---
        self._l1_ratio: float = float(self.params.get("alpha", 0.5))    # 0=ridge-like, 1=lasso-like
        self._alpha: float = float(self.params.get("lambda", 1e-2))     # regularization strength
        self._seed: int = int(self.params.get("seed", 42))

        # Robust defaults
        self._max_iter: int = int(self.params.get("max_iter", 2000))
        self._tol: float = float(self.params.get("tol", 1e-4))
        self._fit_intercept: bool = bool(self.params.get("fit_intercept", True))
        self._selection: str = str(self.params.get("selection", "random"))  # often faster for large p

        self._reg = ElasticNet(
            alpha=self._alpha,
            l1_ratio=self._l1_ratio,
            max_iter=self._max_iter,
            tol=self._tol,
            random_state=self._seed,
            fit_intercept=self._fit_intercept,
            selection=self._selection,
        )

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        """Replace non-finite values with 0.0."""
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit z-standardization on X (train) and apply it (stores mu/sigma)."""
        X = self._clean(X)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)

        sigma_safe = sigma.copy()
        sigma_safe[sigma_safe == 0.0] = 1.0  # constant features

        self._mu_ = mu
        self._sigma_ = sigma_safe

        return (X - mu) / sigma_safe

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        """Apply stored z-standardization to new data."""
        if self._mu_ is None or self._sigma_ is None:
            raise RuntimeError("Scaler parameters are not fitted. Call fit() first.")
        X = self._clean(X)

        if X.shape[1] != self._mu_.shape[0]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match fitted scaler ({self._mu_.shape[0]})."
            )

        return (X - self._mu_) / self._sigma_

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Capture feature names when available
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_np = X.values
        elif hasattr(X, "columns"):
            self._feature_names = list(X.columns)
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self._feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

        y_np = np.asarray(y, dtype=float).ravel()

        # Train-only z-standardization
        X_std = self._standardize_fit(X_np)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            self._reg.fit(X_std, y_np, sample_weight=sample_weight)
        else:
            self._reg.fit(X_std, y_np)

        # Importances = |coefficients| (on standardized feature space)
        self._importances = np.abs(self._reg.coef_)
        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model is not fitted.")
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_std = self._standardize_apply(X_np)
        return self._reg.predict(X_std)

    def predict_one(self, x_row):
        """Predict a single row."""
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        if self._importances is None:
            return {}
        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances))]
        return dict(zip(names, self._importances))

    # Optional: useful for diagnostics
    def get_coef(self) -> Optional[np.ndarray]:
        """Return coefficients in standardized feature space."""
        return None if self._reg is None else self._reg.coef_.copy()

    def get_intercept(self) -> Optional[float]:
        """Return intercept."""
        return None if self._reg is None else float(self._reg.intercept_)
