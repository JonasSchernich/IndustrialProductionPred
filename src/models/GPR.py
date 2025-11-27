# src/models/GPR.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C


class ForecastModel:
    """
    GaussianProcessRegressor-Wrapper im ET/EN/LGBM-Stil:
    - .fit(X, y, sample_weight=None)
    - .predict(X)
    - .predict_one(x_row)
    - .get_feature_importances()  (Pseudo-Importances via ARD-Längenskalen)

    Wichtige Punkte:
      * Dieses Modell führt intern eine train-only Z-Standardisierung von X durch
        (Mittelwert/Std aus dem Trainingsset), BEVOR der GPR gefittet wird.
      * NaN/Inf werden auf 0.0 gesetzt.
      * Feature-"Importances": bei ARD-RBF/Matern: 1 / length_scale^2 (normalisiert)
        bezogen auf den STANDARDISIERTEN Feature-Space.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._reg: Optional[GaussianProcessRegressor] = None
        self._backend_name: str = "gpr"
        self._feature_names: Optional[List[str]] = None

        self._ard: bool = bool(self.params.get("ard", True))
        self._kernel_type: str = str(self.params.get("kernel", "rbf")).lower()
        self._length_scale_init: float = float(self.params.get("length_scale_init", 1.0))
        self._nu: float = float(self.params.get("nu", 1.5))  # nur für Matern
        self._noise_alpha_base: float = float(self.params.get("noise_alpha", 1e-6))
        self._normalize_y: bool = bool(self.params.get("normalize_y", True))
        self._n_restarts_optimizer: int = int(self.params.get("n_restarts_optimizer", 2))
        self._optimizer: Optional[str] = self.params.get("optimizer", "fmin_l_bfgs_b")
        self._seed: Optional[int] = int(self.params.get("seed", 42))

        # Wird nach dem Fit gefüllt (falls ARD):
        self._last_length_scales_: Optional[np.ndarray] = None

        # Skalierungsparameter (train-only Z-Standardisierung)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit Z-Standardisierung auf X (Train) und wende sie an.
        """
        X = self._clean(X)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)

        sigma_safe = sigma.copy()
        sigma_safe[sigma_safe == 0.0] = 1.0  # konstante Features

        self._mu_ = mu
        self._sigma_ = sigma_safe

        return (X - mu) / sigma_safe

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        """
        Wende gespeicherte Z-Standardisierung auf neue Daten an.
        """
        if self._mu_ is None or self._sigma_ is None:
            raise RuntimeError("Scaler parameters not fitted. Call fit() first.")
        X = self._clean(X)

        if X.shape[1] != self._mu_.shape[0]:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match "
                f"fitted scaler ({self._mu_.shape[0]})."
            )

        return (X - self._mu_) / self._sigma_

    def _build_kernel(self, n_features: int):
        # ARD: length_scale ist Vektor der Länge p; sonst Skalar
        if self._ard:
            ls = np.full(n_features, float(self._length_scale_init), dtype=float)
        else:
            ls = float(self._length_scale_init)

        if self._kernel_type == "rbf":
            base = RBF(length_scale=ls)
        elif self._kernel_type == "matern":
            base = Matern(length_scale=ls, nu=self._nu)
        else:
            raise ValueError("kernel must be 'rbf' or 'matern'")

        # Konstante * Base; Rauschanteil rein über alpha (hetero) statt WhiteKernel
        return C(1.0, (1e-3, 1e3)) * base

    def _make_reg(self, n_features: int, alpha_vec: Optional[np.ndarray] = None) -> GaussianProcessRegressor:
        kernel = self._build_kernel(n_features)
        reg = GaussianProcessRegressor(
            kernel=kernel,
            alpha=(self._noise_alpha_base if alpha_vec is None else alpha_vec),
            normalize_y=self._normalize_y,
            optimizer=self._optimizer,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._seed,
        )
        return reg

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Feature-Namen organisieren
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
        n, p = X_np.shape

        # Train-only Z-Standardisierung
        X_std = self._standardize_fit(X_np)

        # Heteroskedastische alpha via sample_weight
        alpha_vec = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).ravel()
            if w.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")
            eps = 1e-12
            alpha_vec = self._noise_alpha_base / np.clip(w, eps, 1.0)

        # Reg mit passendem alpha erzeugen und fitten
        self._reg = self._make_reg(p, alpha_vec=alpha_vec)
        self._reg.fit(X_std, y_np)

        # Längenskalen (für ARD) extrahieren -> Pseudo-Importances
        self._last_length_scales_ = self._extract_length_scales(self._reg)

        return self

    @staticmethod
    def _extract_length_scales(reg: GaussianProcessRegressor) -> Optional[np.ndarray]:
        # Erwartet: Kernel ~ Constant * (RBF|Matern)
        try:
            k = reg.kernel_
            # Bei C(1.0) * Base sind die Faktoren in .k1 (Constant) und .k2 (Base)
            base = getattr(k, "k2", None)
            if base is None:
                base = k  # falls kein Produkt
            if hasattr(base, "length_scale"):
                ls = np.asarray(base.length_scale, dtype=float)
                return ls.copy()
        except Exception:
            pass
        return None

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_std = self._standardize_apply(X_np)
        return self._reg.predict(X_std)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        # Pseudo-Importances aus ARD-Längenskalen: 1 / ls^2 (normalisiert)
        ls = self._last_length_scales_
        if ls is None:
            return {}
        ls = np.asarray(ls, dtype=float)
        if ls.ndim == 0:
            # kein ARD (Skalar) -> nicht sinnvoll aufteilen
            return {}

        imp = 1.0 / (ls ** 2 + 1e-12)
        if np.all(imp <= 0):
            return {}
        imp = (imp / np.sum(imp)).astype(float)

        names = self._feature_names or [f"feature_{i}" for i in range(len(imp))]
        return dict(zip(names, imp.tolist()))
