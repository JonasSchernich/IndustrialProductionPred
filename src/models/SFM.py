# src/models/SFM.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge


class ForecastModel:
    """
    Static Factor Model (SFM): PCA-Faktoren + lineare Regression
    API analog zu ET/EN/LGBM:
      - .fit(X, y, sample_weight=None)
      - .predict(X)
      - .predict_one(x_row)
      - .get_feature_importances()

    Annahmen / Hinweise:
      * Upstream sind Features train-only standardisiert (Z-Score), wie in ET/EN/LGBM.
      * NaN/Inf werden als Sicherheitsnetz auf 0 gesetzt (wie ET/EN/LGBM).
      * PCA wird auf dem Trainings-X gelernt (kein Leakage).
      * sample_weight wirkt auf die Regression (WLS), nicht auf die PCA (wie bei FE/DR im Projekt üblich).

    Hyperparameter (params):
      - n_factors      : int        Anzahl der PCA-Faktoren (k)
      - reg            : {"ols","ridge"}  Regressor-Typ
      - ridge_alpha    : float      Ridge-Penalty (falls reg="ridge")
      - svd_solver     : {"auto","full","randomized"}  PCA-Solver
      - fit_intercept  : bool       Intercept in der Regression
      - seed           : int        Randomstate für PCA (nur relevant bei randomized/arpack)
      - (ignoriert werden z.B. n_features_to_use, corr_* etc. — kommen aus dem Tuning-Grid)

    Feature-Importances:
      - Aus den impliziten Koeffizienten auf Feature-Ebene:
        y_hat = (X_std @ components_.T) @ beta  => w = components_.T @ beta
        Importances = |w| pro Original-Feature.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._backend_name: str = "sfm"

        # --- Hyperparameter / Defaults ---
        self._n_factors: int = int(self.params.get("n_factors", 8))
        self._reg_type: str = str(self.params.get("reg", "ols")).lower()
        self._ridge_alpha: float = float(self.params.get("ridge_alpha", 0.0))
        self._fit_intercept: bool = bool(self.params.get("fit_intercept", True))

        self._svd_solver: str = str(self.params.get("svd_solver", "auto"))
        self._seed: int = int(self.params.get("seed", 42))

        # --- Fitted Objekte ---
        self._pca: Optional[PCA] = None
        self._reg = None  # LinearRegression oder Ridge
        self._feature_names: Optional[List[str]] = None
        self._importances: Optional[np.ndarray] = None  # |w| auf Feature-Ebene
        self._coef_features: Optional[np.ndarray] = None  # w auf Feature-Ebene
        self._coef_factors: Optional[np.ndarray] = None   # beta auf Faktor-Ebene
        self._intercept_: Optional[float] = None

        # Regressor aufsetzen
        if self._reg_type == "ridge":
            self._reg = Ridge(alpha=self._ridge_alpha, fit_intercept=self._fit_intercept, random_state=self._seed)
        elif self._reg_type == "ols":
            self._reg = LinearRegression(fit_intercept=self._fit_intercept)
        else:
            raise ValueError("reg must be 'ols' or 'ridge'.")

        # PCA vorbereiten
        self._pca = PCA(
            n_components=self._n_factors,
            svd_solver=self._svd_solver,
            random_state=self._seed,
        )

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Feature-Namen speichern
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_np = X.values
        elif hasattr(X, "columns"):
            self._feature_names = list(X.columns)
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self._feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

        X_np = self._clean(X_np)
        y_np = np.asarray(y, dtype=float).ravel()

        # n_factors an Daten begrenzen (robust)
        max_k = max(1, min(X_np.shape[1], X_np.shape[0]) - 1)
        k = int(np.clip(self._n_factors, 1, max_k))
        if k != self._n_factors:
            # Falls Grid zu hoch gegriffen hat, stillschweigend k kappen
            self._n_factors = k
            # PCA neu aufsetzen mit gekapptem k
            self._pca = PCA(
                n_components=self._n_factors,
                svd_solver=self._svd_solver,
                random_state=self._seed,
            )

        # PCA auf Train-X
        F_tr = self._pca.fit_transform(X_np)  # (n, k)

        # Regression auf Faktoren (WLS, falls sample_weight)
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")

        # Fit
        if sw is None:
            self._reg.fit(F_tr, y_np)
        else:
            self._reg.fit(F_tr, y_np, sample_weight=sw)

        # Koeffizienten speichern
        if hasattr(self._reg, "coef_"):
            beta = np.asarray(self._reg.coef_, dtype=float).ravel()  # (k,)
        else:
            # sollte nie passieren bei LinearRegression/Ridge
            beta = np.zeros(self._n_factors, dtype=float)

        self._coef_factors = beta.copy()
        self._intercept_ = float(getattr(self._reg, "intercept_", 0.0))

        # Implizite Feature-Koeffizienten: w = components_.T @ beta
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
        X_np = self._clean(X_np)

        F = self._pca.transform(X_np)
        return self._reg.predict(F)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        if self._importances is None:
            return {}
        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances))]
        return dict(zip(names, self._importances.tolist()))

    # Optional: nützlich für Diagnose
    def get_coef_features(self) -> Optional[np.ndarray]:
        """Implizite Koeffizienten auf Feature-Ebene (w)."""
        return None if self._coef_features is None else self._coef_features.copy()

    def get_coef_factors(self) -> Optional[np.ndarray]:
        """Koeffizienten auf Faktoren-Ebene (beta)."""
        return None if self._coef_factors is None else self._coef_factors.copy()

    def get_intercept(self) -> Optional[float]:
        return self._intercept_
