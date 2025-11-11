# src/models/SVR.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.svm import SVR as _SVR


class ForecastModel:
    """
    Support Vector Regression (SVR) – Wrapper im ET/EN/LGBM-Stil
    ------------------------------------------------------------
    API:
      - .fit(X, y, sample_weight=None)
      - .predict(X)
      - .predict_one(x_row)
      - .get_feature_importances()  -> bei kernel='linear' via |w|, sonst {}

    Hinweise:
      * Upstream sollten Features (train-only) standardisiert sein (Z-Score).
      * NaN/Inf werden defensiv auf 0.0 gesetzt (analog ET/EN/LGBM).
      * sample_weight wird an sklearn.SVR.fit weitergereicht.
      * Für kernel='linear' werden Pseudo-Importances als |w| berechnet:
            w = dual_coef_ @ support_vectors_
        Für nichtlineare Kernel werden keine Importances zurückgegeben ({}).

    Wichtige Hyperparameter (gleich zur sklearn-Notation):
      - kernel: {'rbf','linear','poly','sigmoid'}
      - C: float
      - epsilon: float
      - gamma: {'scale','auto'} oder float
      - degree: int (für 'poly')
      - coef0: float (für 'poly'/'sigmoid')
      - shrinking: bool
      - tol: float
      - max_iter: int (<=0 bedeutet 'unbegrenzt' in sklearn)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._reg: Optional[_SVR] = None
        self._backend_name: str = "svr"
        self._feature_names: Optional[List[str]] = None
        self._importances: Optional[np.ndarray] = None

        # --- robuste Defaults (sklearn-kompatibel) ---
        self._kernel: str      = str(self.params.get("kernel", "rbf"))
        self._C: float         = float(self.params.get("C", 1.0))
        self._epsilon: float   = float(self.params.get("epsilon", 0.1))
        self._gamma: Any       = self.params.get("gamma", "scale")
        self._degree: int      = int(self.params.get("degree", 3))
        self._coef0: float     = float(self.params.get("coef0", 0.0))
        self._shrinking: bool  = bool(self.params.get("shrinking", True))
        self._tol: float       = float(self.params.get("tol", 1e-3))
        self._max_iter: int    = int(self.params.get("max_iter", -1))  # -1 = unbegrenzt

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

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Feature-Namen erfassen (analog ET/EN/LGBM)
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

        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")

        self._reg.fit(X_np, y_np, sample_weight=sw)

        # Feature-"Importances" (nur linearer Kernel)
        self._importances = None
        if self._kernel == "linear":
            try:
                # w = dual_coef_ @ support_vectors_  -> (1, n_support) @ (n_support, n_features)
                w = self._reg.dual_coef_.dot(self._reg.support_vectors_)
                w = np.asarray(w).ravel()
                if w.shape[0] == X_np.shape[1]:
                    self._importances = np.abs(w)
            except Exception:
                self._importances = None

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
        if self._importances is None:
            return {}
        names = self._feature_names or [f"feature_{i}" for i in range(len(self._importances))]
        return dict(zip(names, self._importances.tolist()))

    # Optional: Diagnosen
    def get_linear_weights(self) -> Optional[np.ndarray]:
        """Nur sinnvoll bei kernel='linear'. Liefert w (oder None)."""
        return None if self._importances is None else self._importances.copy()

    def get_intercept(self) -> Optional[float]:
        return None if self._reg is None else float(self._reg.intercept_.ravel()[0])
