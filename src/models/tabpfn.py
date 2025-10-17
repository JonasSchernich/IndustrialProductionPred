# src/models/tabpfn.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

class ForecastModel:
    """
    TabPFN-Regressor als Drop-in für die Tuning-Pipeline.
    - .fit(X, y), .predict_one(x_row)
    - optionale HPs: use_gpu (True/False), posterior_samples (int)
    - unbekannte Keys werden best-effort an TabPFNRegressor durchgereicht (fallback-sicher)
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self._reg = None
        self._backend_name = "tabpfn"
        self._use_gpu = bool(self.params.pop("use_gpu", False))
        self._posterior_samples = int(self.params.pop("posterior_samples", 8))

    def get_name(self) -> str:
        dev = "cuda" if self._use_gpu else "cpu"
        return f"tabpfn[{dev}]"

    def _build_regressor(self):
        try:
            from tabpfn import TabPFNRegressor
        except Exception as e:
            raise ImportError("TabPFN nicht installiert. Bitte `pip install tabpfn torch` ausführen.") from e

        # Device
        device = "cpu"
        if self._use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"

        base_kwargs: Dict[str, Any] = {
            "device": device,
            "ignore_pretraining_limits": True,
        }
        # Unbekannte Keys defensiv durchreichen
        extra = dict(self.params)
        try:
            return TabPFNRegressor(**{**base_kwargs, **extra})
        except TypeError:
            return TabPFNRegressor(**base_kwargs)

    @staticmethod
    def _clean(X):
        X = np.asarray(X, dtype=float)
        # NaN/Inf robust behandeln (TabPFN erwartet numerische Features)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def fit(self, X, y):
        X = self._clean(X)
        y = np.asarray(y, dtype=float).ravel()
        self._reg = self._build_regressor()
        self._reg.fit(X, y)
        return self

    def predict(self, X):
        if self._reg is None:
            raise RuntimeError("Model not fitted.")
        X = self._clean(X)
        try:
            return self._reg.predict(X, posterior_samples=self._posterior_samples)
        except TypeError:
            return self._reg.predict(X)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])
