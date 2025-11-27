from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.svm import SVR as _SVR, NuSVR as _NuSVR


class ForecastModel:
    """
    Support Vector Regression (SVR) – Wrapper.

    Workflow:
      1. Train-only Z-Standardisierung von X.
      2. SVR auf standardisierten Daten.

    Varianten:
      - epsilon-SVR (sklearn.svm.SVR) mit Parameter 'epsilon'
      - nu-SVR      (sklearn.svm.NuSVR) mit Parameter 'nu'

    Feature Importances:
      - Nur verfügbar, wenn kernel='linear'.
      - Werden als |w| (Standardized Coefficients) berechnet.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._reg: Optional[_SVR] = None
        self._backend_name: str = "svr"
        self._feature_names: Optional[List[str]] = None
        self._importances: Optional[np.ndarray] = None

        # Skalierungsparameter (train-only Z-Standardisierung)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None

        # --- robuste Defaults (sklearn-kompatibel) ---
        self._variant: str = str(self.params.get("variant", "epsilon")).lower()
        self._kernel: str = str(self.params.get("kernel", "rbf")).lower()
        self._C: float = float(self.params.get("C", 1.0))
        self._epsilon: float = float(self.params.get("epsilon", 0.1))
        self._nu: float = float(self.params.get("nu", 0.1))
        self._gamma: Any = self.params.get("gamma", "scale")
        self._degree: int = int(self.params.get("degree", 3))
        self._coef0: float = float(self.params.get("coef0", 0.0))
        self._shrinking: bool = bool(self.params.get("shrinking", True))
        self._tol: float = float(self.params.get("tol", 1e-3))
        self._max_iter: int = int(self.params.get("max_iter", -1))  # -1 = unbegrenzt
        self._seed: int = int(self.params.get("seed", 42))

        # HINWEIS: SVR/NuSVR haben keinen 'random_state'-Parameter,
        # aber wir speichern ihn für Konsistenz.

        # --- Regressor aufsetzen (epsilon-SVR oder nu-SVR) ---
        if self._variant == "nu":
            # nu-SVR: nutzt 'nu' statt 'epsilon'
            self._reg = _NuSVR(
                kernel=self._kernel,
                C=self._C,
                nu=self._nu,
                gamma=self._gamma,
                degree=self._degree,
                coef0=self._coef0,
                shrinking=self._shrinking,
                tol=self._tol,
                max_iter=self._max_iter,
            )
        else:
            # Default: epsilon-SVR
            self._variant = "epsilon"
            self._reg = _SVR(
                kernel=self._kernel,
                C=self._C,
                epsilon=self._epsilon,
                gamma=self._gamma,
                degree=self._degree,
                coef0=self._coef0,
                shrinking=self._shrinking,
                tol=self._tol,
                max_iter=self._max_iter,
            )

    def get_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit Z-Standardisierung auf X (Train) und wende sie an."""
        X = self._clean(X)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)

        sigma_safe = sigma.copy()
        sigma_safe[sigma_safe == 0.0] = 1.0  # konstante Features

        self._mu_ = mu
        self._sigma_ = sigma_safe

        return (X - mu) / sigma_safe

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        """Wende gespeicherte Z-Standardisierung auf neue Daten an."""
        if self._mu_ is None or self._sigma_ is None:
            raise RuntimeError("Scaler parameters not fitted. Call fit() first.")
        X = self._clean(X)

        if X.shape[1] != self._mu_.shape[0]:
            raise ValueError(f"Feature mismatch: X({X.shape[1]}) vs fitted({self._mu_.shape[0]}).")

        return (X - self._mu_) / self._sigma_

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Metadata
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

        # 1. Standardisierung
        X_std = self._standardize_fit(X_np)

        # 2. Fit
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")

        self._reg.fit(X_std, y_np, sample_weight=sw)

        # 3. Importances (nur bei linear kernel)
        self._importances = None
        if self._kernel == "linear":
            if hasattr(self._reg, "coef_"):
                self._importances = np.abs(self._reg.coef_.ravel())

        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")

        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        # Standardisierung anwenden
        X_std = self._standardize_apply(X_np)
        return self._reg.predict(X_std)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        if self._importances is None:
            return {}
        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances))]
        return dict(zip(names, self._importances.tolist()))

    def get_linear_weights(self) -> Optional[np.ndarray]:
        """Nur sinnvoll bei kernel='linear'. Liefert |w| (oder None) bzgl. standardisiertem X."""
        return None if self._importances is None else self._importances.copy()

    def get_intercept(self) -> Optional[float]:
        return None if self._reg is None else float(self._reg.intercept_.ravel()[0])
