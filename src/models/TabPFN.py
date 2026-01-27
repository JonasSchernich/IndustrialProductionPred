from __future__ import annotations
from typing import Any, Dict, Optional, List
import warnings
import numpy as np
import pandas as pd

# Use torch only for device routing (cpu/cuda/mps)
try:
    import torch
except Exception as e:
    raise ImportError(
        "PyTorch is required to determine the compute device (cpu/cuda/mps) for TabPFN."
    ) from e

try:
    # Official sklearn-compatible API of TabPFN v2.x
    from tabpfn import TabPFNRegressor
except Exception as e:
    raise ImportError(
        "TabPFN (tabpfn) is not installed or not importable. "
        "Install it, e.g.: pip install tabpfn"
    ) from e


def _metal_available() -> bool:
    """Return True if the pyobjc Metal framework is importable (safer MPS routing on macOS)."""
    try:
        from Metal import MTLCreateSystemDefaultDevice  # noqa: F401
        return True
    except Exception:
        return False


def _select_device(use_gpu: bool, explicit_device: Optional[str] = None) -> str:
    """
    Select compute device.
    - If explicit_device is set: 'cpu'/'cuda'/'mps'
    - Else: 'cpu'; if use_gpu=True -> prefer cuda, else mps.
    """
    # Explicit override
    if explicit_device:
        dev = explicit_device.lower()
        if dev == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            warnings.warn("Explicit 'cuda' requested, but CUDA is not available -> falling back to 'cpu'.", stacklevel=1)
            return "cpu"
        if dev == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and _metal_available():
                return "mps"
            warnings.warn("Explicit 'mps' requested, but MPS/Metal is not usable -> falling back to 'cpu'.", stacklevel=1)
            return "cpu"
        if dev == "cpu":
            return "cpu"
        warnings.warn(f"Unknown device '{explicit_device}', falling back to 'cpu'.", stacklevel=1)
        return "cpu"

    # Automatic selection
    if not use_gpu:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and _metal_available():
        return "mps"

    warnings.warn(
        "GPU requested, but neither CUDA nor (safely usable) MPS is available -> using 'cpu'.",
        stacklevel=1,
    )
    return "cpu"


class ForecastModel:
    """
    TabPFN wrapper.

    Notes:
    - Pre-trained Transformer for tabular data.
    - Uses train-only z-standardization.
    - Sample weights are ignored (not supported).
    - Automatic device selection (CPU/CUDA/MPS).

    Params:
    - 'use_gpu': bool -> auto-select GPU
    - 'device': str -> force 'cpu', 'cuda', or 'mps'
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})
        self._backend_name: str = "tabpfn"
        self._feature_names: Optional[List[str]] = None
        self._model: Optional[TabPFNRegressor] = None

        # Scaling params (train-only z-standardization)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None

        # Config from params
        self._use_gpu: bool = bool(self.params.get("use_gpu", False))
        self._explicit_device: Optional[str] = self.params.get("device")
        self._n_jobs: int = int(self.params.get("n_jobs", 1))
        self._seed: int = int(self.params.get("seed", 42))

        # Select device
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
        """Convert to float array and replace NaN/inf with 0."""
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit z-standardization on X (train) and apply it."""
        X = self._clean(X)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)

        sigma_safe = sigma.copy()
        sigma_safe[sigma_safe == 0.0] = 1.0

        self._mu_ = mu
        self._sigma_ = sigma_safe

        return (X - mu) / sigma_safe

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        """Apply stored z-standardization to new data."""
        if self._mu_ is None or self._sigma_ is None:
            raise RuntimeError("Scaler parameters are not fitted. Call fit() first.")
        X = self._clean(X)

        if X.shape[1] != self._mu_.shape[0]:
            raise ValueError(f"Feature mismatch: X has {X.shape[1]} features, fitted has {self._mu_.shape[0]}.")

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

        # 1) Standardize
        X_std = self._standardize_fit(X_np)

        # 2) Sample weights (not supported)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if sw.shape[0] != y_np.shape[0]:
                raise ValueError("sample_weight length must match y.")
            warnings.warn("sample_weight is ignored by TabPFN.", stacklevel=1)

        if self._model is None:
            raise RuntimeError("Internal error: TabPFNRegressor is not initialized.")

        self._model.fit(X_std, y_np)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model is not fitted.")
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_std = self._standardize_apply(X_np)
        yhat = self._model.predict(X_std)
        return np.asarray(yhat, dtype=float)

    def predict_one(self, x_row):
        """Predict a single row."""
        x = np.asarray(x_row).reshape(1, -1)
        return float(self.predict(x)[0])

    def get_feature_importances(self) -> Dict[str, float]:
        """TabPFN does not provide feature importances."""
        return {}
