from __future__ import annotations
from typing import Any, Dict, Optional, List
import warnings
import numpy as np
import pandas as pd

# Torch nur fürs Device-Routing verwenden
try:
    import torch
except Exception as e:
    raise ImportError(
        "PyTorch wird benötigt, um das Rechen-Device (cpu/cuda/mps) für TabPFN zu bestimmen."
    ) from e

try:
    # Offizielle sklearn-kompatible API von TabPFN v2.x
    from tabpfn import TabPFNRegressor
except Exception as e:
    raise ImportError(
        "TabPFN (tabpfn) ist nicht installiert oder nicht importierbar. "
        "Installiere es z.B. mit: pip install tabpfn"
    ) from e


def _metal_available() -> bool:
    """
    Prüft, ob das pyobjc Metal Framework importierbar ist (für MPS-Speicherschätzer).
    Auf macOS ist das für sicheres MPS-Routing hilfreich.
    """
    try:
        from Metal import MTLCreateSystemDefaultDevice  # noqa: F401
        return True
    except Exception:
        return False


def _select_device(use_gpu: bool, explicit_device: Optional[str] = None) -> str:
    """
    Wählt das Device:
      - Wenn explicit_device gesetzt: 'cpu'/'cuda'/'mps'
      - Sonst: cpu; bei use_gpu=True -> cuda bevorzugt, sonst mps.
    """
    # 1) Explizit erzwungen?
    if explicit_device:
        dev = explicit_device.lower()
        if dev == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            warnings.warn("Explizit 'cuda' gewählt, aber keine CUDA verfügbar -> fallback auf 'cpu'.", stacklevel=1)
            return "cpu"
        if dev == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and _metal_available():
                return "mps"
            warnings.warn("Explizit 'mps' gewählt, aber MPS/Metal nicht nutzbar -> fallback auf 'cpu'.", stacklevel=1)
            return "cpu"
        if dev == "cpu":
            return "cpu"
        warnings.warn(f"Unbekanntes Device '{explicit_device}', fallback auf 'cpu'.", stacklevel=1)
        return "cpu"

    # 2) Automatik
    if not use_gpu:
        return "cpu"

    # GPU-Autowahl
    if torch.cuda.is_available():
        return "cuda"

    # MPS nur, wenn verfügbar und Metal importierbar ist
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and _metal_available():
        return "mps"

    warnings.warn("GPU angefragt, aber weder CUDA noch (sicher nutzbares) MPS vorhanden -> 'cpu' verwendet.",
                  stacklevel=1)
    return "cpu"


class ForecastModel:
    """
    TabPFN-Wrapper.

    Besonderheiten:
      - Pre-trained Transformer für Tabular Data.
      - Führt intern train-only Z-Standardisierung durch.
      - Sample Weights werden ignoriert (nicht unterstützt).
      - Automatische Device-Wahl (CPU/CUDA/MPS).

    Hyperparameter:
      - 'use_gpu': bool -> Autowahl GPU
      - 'device': str -> Erzwingt 'cpu', 'cuda', 'mps'
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._backend_name: str = "tabpfn"
        self._feature_names: Optional[List[str]] = None
        self._model: Optional[TabPFNRegressor] = None

        # Skalierungsparameter (train-only Z-Standardisierung)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None

        # --- Konfiguration aus params ---
        self._use_gpu: bool = bool(self.params.get("use_gpu", False))
        self._explicit_device: Optional[str] = self.params.get("device")
        self._n_jobs: int = int(self.params.get("n_jobs", 1))
        self._seed: int = int(self.params.get("seed", 42))

        # Device wählen
        self._device: str = _select_device(self._use_gpu, self._explicit_device)

        self._model = TabPFNRegressor(
            device=self._device,
            n_jobs=self._n_jobs,
        )

    def get_name(self) -> str:
        return self._backend_name

    def get_device(self) -> str:
        return self._device

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
        sigma_safe[sigma_safe == 0.0] = 1.0

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

        # 2. Sample Weights (nicht unterstützt)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")
            warnings.warn("sample_weight wird von TabPFN ignoriert.", stacklevel=1)

        if self._model is None:
            raise RuntimeError("Interner Fehler: TabPFN-Regressor nicht initialisiert.")

        self._model.fit(X_std, y_np)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_std = self._standardize_apply(X_np)
        yhat = self._model.predict(X_std)
        return np.asarray(yhat, dtype=float)

    def predict_one(self, x_row):
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        # TabPFN liefert keine Feature Importances
        return {}