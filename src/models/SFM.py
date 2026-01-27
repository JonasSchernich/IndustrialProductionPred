from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge


class ForecastModel:
    """
    Static Factor Model (SFM): PCA factors + linear regression.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._backend_name: str = "sfm"

        # Hyperparameters / defaults
        self._n_factors: int = int(self.params.get("n_factors", 8))
        self._reg_type: str = str(self.params.get("reg", "ols")).lower()
        self._ridge_alpha: float = float(self.params.get("ridge_alpha", 0.0))
        self._fit_intercept: bool = bool(self.params.get("fit_intercept", True))

        self._svd_solver: str = str(self.params.get("svd_solver", "auto"))
        self._seed: int = int(self.params.get("seed", 42))

        # Fitted objects
        self._pca: Optional[PCA] = None
        self._reg = None  # LinearRegression or Ridge
        self._feature_names: Optional[List[str]] = None
        self._importances: Optional[np.ndarray] = None   # |w| on feature level
        self._coef_features: Optional[np.ndarray] = None  # w on feature level
        self._coef_factors: Optional[np.ndarray] = None   # beta on factor level
        self._intercept_: Optional[float] = None

        # Scaling params (train-only z-standardization)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None

        # Build regressor
        if self._reg_type == "ridge":
            self._reg = Ridge(alpha=self._ridge_alpha, fit_intercept=self._fit_intercept, random_state=self._seed)
        elif self._reg_type == "ols":
            self._reg = LinearRegression(fit_intercept=self._fit_intercept)
        else:
            raise ValueError("reg must be 'ols' or 'ridge'.")

        # Prepare PCA
        self._pca = PCA(
            n_components=self._n_factors,
            svd_solver=self._svd_solver,
            random_state=self._seed,
        )

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        """Convert to float array and replace NaN/Inf with 0."""
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit z-standardization on X (train) and apply it; store mu/sigma for predict()."""
        X = self._clean(X)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)

        # Guard against division by zero: sigma == 0 -> 1
        sigma_safe = sigma.copy()
        sigma_safe[sigma_safe == 0.0] = 1.0

        self._mu_ = mu
        self._sigma_ = sigma_safe

        return (X - mu) / sigma_safe

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        """Apply stored z-standardization to new data."""
        if self._mu_ is None or self._sigma_ is None:
            raise RuntimeError("Scaler parameters not fitted. Call fit() first.")
        X = self._clean(X)

        if X.shape[1] != self._mu_.shape[0]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match "
                f"fitted scaler ({self._mu_.shape[0]})."
            )

        return (X - self._mu_) / self._sigma_

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Store feature names when available
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

        # Clamp n_factors to valid range (robust against over-sized grids)
        max_k = max(1, min(X_np.shape[1], X_np.shape[0]) - 1)
        k = int(np.clip(self._n_factors, 1, max_k))
        if k != self._n_factors:
            self._n_factors = k
            self._pca = PCA(
                n_components=self._n_factors,
                svd_solver=self._svd_solver,
                random_state=self._seed,
            )

        # Standardize using train statistics
        X_std = self._standardize_fit(X_np)

        # Fit PCA on standardized train X
        F_tr = self._pca.fit_transform(X_std)  # (n, k)

        # Fit regression on factors (WLS if sample_weight is provided)
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")

        if sw is None:
            self._reg.fit(F_tr, y_np)
        else:
            self._reg.fit(F_tr, y_np, sample_weight=sw)

        # Store coefficients
        if hasattr(self._reg, "coef_"):
            beta = np.asarray(self._reg.coef_, dtype=float).ravel()  # (k,)
        else:
            beta = np.zeros(self._n_factors, dtype=float)

        self._coef_factors = beta.copy()
        self._intercept_ = float(getattr(self._reg, "intercept_", 0.0))

        # Implicit feature coefficients: w = components_.T @ beta
        # components_.shape = (k, p)  => w.shape = (p,)
        comps = np.asarray(self._pca.components_, dtype=float)
        w = comps.T @ beta
        self._coef_features = w
        self._importances = np.abs(w)

        return self

    def predict(self, X):
        if self._pca is None or self._reg is None:
            raise RuntimeError("Model not fitted.")

        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        # Apply the same standardization as in training
        X_std = self._standardize_apply(X_np)
        F = self._pca.transform(X_std)
        return self._reg.predict(F)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        if self._importances is None:
            return {}
        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances))]
        return dict(zip(names, self._importances.tolist()))

    # Optional: useful for diagnostics
    def get_coef_features(self) -> Optional[np.ndarray]:
        """Implicit feature-level coefficients (w, w.r.t. standardized X)."""
        return None if self._coef_features is None else self._coef_features.copy()

    def get_coef_factors(self) -> Optional[np.ndarray]:
        """Factor-level coefficients (beta)."""
        return None if self._coef_factors is None else self._coef_factors.copy()

    def get_intercept(self) -> Optional[float]:
        return self._intercept_
