# src/models/EN.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit


class ForecastModel:
    """
    ElasticNetCV-Regressor als Drop-in für die Tuning-Pipeline.
    - .fit(X, y, sample_weight=None)
    - .predict_one(x_row)
    - HPs (aus dem Grid):
        - "alpha" (Thesis-Def): Der L1-Misch-Parameter (wird auf sklearns l1_ratio gemappt).
        - "cv" (optional): Anzahl der Folds für TimeSeriesSplit.
        - "n_alphas" (optional): Anzahl der Lambda-Werte (sklearn 'alpha'), die CV testet.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self._reg = None
        self._backend_name = "elastic_net"

        # --- WICHTIGES Mapping (Thesis-Def -> Sklearn-Def) ---
        # Ihr 'alpha' (Mischung) ist sklearns 'l1_ratio'.
        self._l1_ratio = float(self.params.get("alpha", 0.5))

        # Ihr 'lambda' (Strafe) wird von ElasticNetCV (als 'alpha') automatisch gefunden.
        # --- Ende Mapping ---

        # CV-Parameter für die Lambda-Suche
        self._cv_splits = int(self.params.get("cv", 5))
        self._n_alphas = int(self.params.get("n_alphas", 100))

    def get_name(self) -> str:
        return f"en[l1_ratio={self._l1_ratio}]"

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def fit(self, X, y, sample_weight=None):
        X = self._clean(X)
        y = np.asarray(y, dtype=float).ravel()

        # Strikte Einhaltung der Zeitreihen-Validierung (wie in Methods/Exp. Design)
        # Wir verwenden TimeSeriesSplit für die interne CV zur Lambda-Findung.
        tscv = TimeSeriesSplit(n_splits=self._cv_splits)

        self._reg = ElasticNetCV(
            l1_ratio=self._l1_ratio,  # Setzt den Misch-Parameter (Ihr 'alpha')
            n_alphas=self._n_alphas,  # Anzahl der Lambdas, die getestet werden
            cv=tscv,  # Zeit-validierte Folds
            random_state=42,
            n_jobs=1,  # Sicherer für verschachtelte Prozesse
            fit_intercept=True
        )

        self._reg.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")
        X = self._clean(X)
        return self._reg.predict(X)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])