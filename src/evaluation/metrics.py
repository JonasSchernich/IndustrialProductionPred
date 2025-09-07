# src/evaluation/metrics.py
from __future__ import annotations
from typing import Tuple, Callable, Optional
import math
import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series

def _to_series(x: ArrayLike, name: str = "v") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x), name=name)

def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, yhat = _to_series(y_true), _to_series(y_pred)
    return float(np.mean(np.abs(y - yhat)))

def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, yhat = _to_series(y_true), _to_series(y_pred)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    y, yhat = _to_series(y_true), _to_series(y_pred)
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - yhat) / denom)) * 100.0)

def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    y, yhat = _to_series(y_true), _to_series(y_pred)
    denom = np.maximum((np.abs(y) + np.abs(yhat)) / 2.0, eps)
    return float(np.mean(np.abs(y - yhat) / denom) * 100.0)

def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    insample: Optional[ArrayLike] = None,
    m: int = 1,
    eps: float = 1e-12,
) -> float:
    y, yhat = _to_series(y_true), _to_series(y_pred)
    ins = y if insample is None else _to_series(insample, name="ins")
    if len(ins) <= m:
        raise ValueError("Not enough data to compute MASE scale.")
    scale = np.mean(np.abs(ins.values[m:] - ins.values[:-m]))
    scale = max(scale, eps)
    return float(np.mean(np.abs(y - yhat)) / scale)

def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike, prev_actual: ArrayLike) -> float:
    y, yhat, prev = _to_series(y_true), _to_series(y_pred), _to_series(prev_actual)
    dy_true = y - prev
    dy_pred = yhat - prev
    return float(np.mean(np.sign(dy_true) == np.sign(dy_pred)))

def _norm_cdf(x: float) -> float:
    # Standardnormal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def dm_test(
    y_true: ArrayLike,
    y_pred_1: ArrayLike,
    y_pred_2: ArrayLike,
    h: int = 1,
    loss: str = "mse",
) -> Tuple[float, float]:
    """
    Diebold-Mariano-Test (asymptotisch) für Prognosen 1 vs. 2.
    Rückgabe: (DM-Statistik, p-Wert, zweiseitig). Fällt auf Normalapproximation zurück, falls SciPy fehlt.
    """
    y = _to_series(y_true)
    e1 = y - _to_series(y_pred_1)
    e2 = y - _to_series(y_pred_2)

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'.")

    d = d.dropna()
    T = len(d)
    if T < 3:
        return np.nan, np.nan

    dbar = float(d.mean())

    # Newey-West Varianzschätzer der Mittelwertschätzung dbar
    q = max(int(h) - 1, 0)
    d_centered = d - dbar
    gamma0 = float(np.mean(d_centered * d_centered))  # Autocov lag 0
    nw = gamma0
    for lag in range(1, q + 1):
        cov = float(np.mean(d_centered.values[lag:] * d_centered.values[:-lag]))
        weight = 1.0 - lag / (q + 1.0)
        nw += 2.0 * weight * cov
    var = nw / T
    if var <= 0:
        return np.nan, np.nan

    dm_stat = dbar / math.sqrt(var)

    # p-Wert: erst t-Verteilung versuchen, sonst Normalapprox.
    try:
        from scipy.stats import t as student_t  # type: ignore
        p_val = 2 * (1 - student_t.cdf(abs(dm_stat), df=T - 1))
    except Exception:
        p_val = 2 * (1 - _norm_cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)

def get_metric(name: str) -> Callable[[ArrayLike, ArrayLike], float]:
    key = name.strip().lower()
    if key == "mae":
        return mae
    if key == "rmse":
        return rmse
    if key == "mape":
        return mape
    if key == "smape":
        return smape
    if key == "mase":
        raise ValueError("MASE requires 'insample' data; compute it explicitly.")
    raise ValueError(f"Unknown metric '{name}'.")
