# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _spec_get(spec: Any, key: str, default=None):
    """Safely read an attribute/key from a spec object or dict."""
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _corr_equal_weight(x: np.ndarray, y: np.ndarray) -> float:
    """Equal-weight (expanding) Pearson correlation."""
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


def _corr_ewma(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    """EWMA Pearson correlation with decay lambda (close to 1 => slower decay)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0

    x = x[-n:]
    y = y[-n:]

    mx = x[0]
    my = y[0]
    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    alpha = 1.0 - lam

    for i in range(1, n):
        xi = x[i]
        yi = y[i]

        dx = xi - mx
        dy = yi - my

        mx += alpha * dx
        my += alpha * dy

        cxx = lam * cxx + alpha * dx * (xi - mx)
        cyy = lam * cyy + alpha * dy * (yi - my)
        cxy = lam * cxy + alpha * dx * (yi - my)

    den = np.sqrt(cxx * cyy)
    return 0.0 if den <= 0.0 else float(cxy / den)


def pw_corr(x: np.ndarray, y: np.ndarray, spec: Any) -> float:
    """
    Pearson correlation using either:
    - expanding (equal weight), or
    - EWMA (mode='ewma'/'ewm', lambda/lam in spec)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0

    mode = _spec_get(spec, "mode", "expanding")
    if mode in ("ewma", "ewm"):
        lam_key1 = _spec_get(spec, "lam")
        lam_key2 = _spec_get(spec, "lambda", 0.98)
        lam = float(lam_key1 if lam_key1 is not None else lam_key2)
        lam = max(1e-6, min(1.0 - 1e-6, lam))
        return _corr_ewma(x, y, lam)

    return _corr_equal_weight(x, y)


# --------------------------------------------------------------------------------------
# Lag mapping (simplified)
# --------------------------------------------------------------------------------------

def select_lags_per_feature(X: pd.DataFrame, L: Tuple[int, ...]) -> Dict[str, List[int]]:
    """Assign all lags in L to every feature column in X."""
    all_lags = sorted(list(L))
    return {col: all_lags for col in X.columns}


# --------------------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------------------

def build_engineered_matrix(X: pd.DataFrame, lag_map: Dict[str, List[int]]) -> pd.DataFrame:
    """Build a lagged feature matrix based on lag_map."""
    out: Dict[str, pd.Series] = {}
    for col, lags in lag_map.items():
        s = X[col]
        for lag in lags:
            out[f"{col}__lag{lag}"] = s.shift(lag).astype(float)
    return pd.DataFrame(out, index=X.index)


# --------------------------------------------------------------------------------------
# screen_k1 (SIS)
# --------------------------------------------------------------------------------------

def screen_k1(
    X_eng: pd.DataFrame,
    y: pd.Series,
    I_t: int,
    corr_spec: Any,
    taus: np.ndarray,
    k1_topk: Optional[int] = 200,
    threshold: Optional[float] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Score features by abs corr with y(t+1) and keep top-k (or threshold)."""
    if len(taus) == 0 or X_eng.shape[1] == 0:
        return [], {}

    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)
    y_mask_base = ~np.isnan(y_next)

    scores: Dict[str, float] = {}
    for col in X_eng.columns:
        xvals = X_eng[col].iloc[taus].to_numpy(dtype=float)
        mask = y_mask_base & (~np.isnan(xvals))

        if np.sum(mask) < 2:
            scores[col] = 0.0
            continue

        scores[col] = abs(pw_corr(xvals[mask], y_next[mask], corr_spec))

    cols = list(X_eng.columns)
    if threshold is not None:
        keep = [c for c in cols if scores.get(c, 0.0) >= float(threshold)]
    else:
        k = int(k1_topk or 0)
        keep = [] if k <= 0 else [c for c, _ in sorted(scores.items(), key=lambda z: z[1], reverse=True)[:k]]

    return keep, scores


# --------------------------------------------------------------------------------------
# redundancy_reduce_greedy
# --------------------------------------------------------------------------------------

def redundancy_reduce_greedy(
    X_sel: pd.DataFrame,
    corr_spec: Any,
    taus: np.ndarray,
    redundancy_param: float = 0.90,
    scores: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Greedy redundancy pruning using pairwise correlation."""
    if X_sel.shape[1] <= 1:
        return list(X_sel.columns)

    order = list(X_sel.columns)
    if scores:
        order.sort(key=lambda c: scores.get(c, 0.0), reverse=True)
    else:
        order.sort()

    X_raw: Dict[str, np.ndarray] = {c: X_sel[c].iloc[taus].to_numpy(dtype=float) for c in order}

    kept: List[str] = []
    for c in order:
        xvals_c = X_raw[c]
        ok = True

        for kcol in kept:
            xvals_k = X_raw[kcol]
            mask = (~np.isnan(xvals_c)) & (~np.isnan(xvals_k))
            if np.sum(mask) < 2:
                continue

            if abs(pw_corr(xvals_c[mask], xvals_k[mask], corr_spec)) > float(redundancy_param):
                ok = False
                break

        if ok:
            kept.append(c)

    return kept


# --------------------------------------------------------------------------------------
# Dimensionality reduction
# --------------------------------------------------------------------------------------

@dataclass
class _DRMap:
    method: str
    cols_: List[str]
    n_components_: int
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None
    pca_components_: Optional[np.ndarray] = None
    pls_model: Optional[object] = None


def _fit_standardizer(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a per-column standardizer (ignoring NaNs)."""
    mu = np.nanmean(X.to_numpy(dtype=float), axis=0)
    sd = np.nanstd(X.to_numpy(dtype=float), axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return mu.astype(float), sd.astype(float)


def _apply_standardizer(X: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply standardization using precomputed mean/std."""
    return (X.to_numpy(dtype=float) - mu.reshape(1, -1)) / sd.reshape(1, -1)


def fit_dr(
    X_tr: pd.DataFrame,
    method: str = "none",
    pca_var_target: float = 0.95,
    pca_kmax: int = 50,
    pls_components: int = 4,
) -> _DRMap:
    """Fit a DR mapping (none/pca/pls) based on training data."""
    cols = list(X_tr.columns)
    n_samples, n_features = X_tr.shape

    m = _DRMap(method=method, cols_=cols, n_components_=n_features)

    if method == "none":
        return m

    mu, sd = _fit_standardizer(X_tr)
    m.scaler_mean, m.scaler_std = mu, sd

    if method == "pca":
        from sklearn.decomposition import PCA

        Z = _apply_standardizer(X_tr, mu, sd)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        k_max_effective = min(n_samples, n_features, int(pca_kmax))

        pca = PCA(svd_solver="full")
        pca.fit(Z)
        evr = pca.explained_variance_ratio_
        k = int(np.searchsorted(np.cumsum(evr), pca_var_target) + 1)
        k = int(min(max(1, k), k_max_effective))

        m.pca_components_ = pca.components_[:k, :].copy()
        m.n_components_ = k
        return m

    if method == "pls":
        from sklearn.cross_decomposition import PLSRegression

        Z = _apply_standardizer(X_tr, mu, sd)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        n, p = Z.shape
        c_eff = int(np.clip(pls_components, 1, max(1, min(p, n - 1))))

        pls = PLSRegression(n_components=c_eff, scale=False)
        m.pls_model = pls
        m.n_components_ = c_eff
        return m

    m.method = "none"
    m.n_components_ = n_features
    return m


def transform_dr(
    m: _DRMap,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    fit_pls: bool = False,
) -> np.ndarray:
    """Transform X using a fitted DR map."""
    if m.method == "none":
        Xc = X.loc[:, m.cols_] if m.cols_ else X
        return np.nan_to_num(Xc.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    if m.scaler_mean is None or m.scaler_std is None:
        raise ValueError("DR map (PCA/PLS) is not initialized correctly (missing scaler).")

    Xc = X.loc[:, m.cols_]
    Z = _apply_standardizer(Xc, m.scaler_mean, m.scaler_std)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    if m.method == "pca":
        V = m.pca_components_
        return Z if V is None else (Z @ V.T).astype(float, copy=False)

    if m.method == "pls":
        from sklearn.cross_decomposition import PLSRegression

        pls: PLSRegression = m.pls_model
        if fit_pls:
            if y is None:
                raise ValueError("PLS transform with fit_pls=True requires y.")
            yv = y.to_numpy(dtype=float).reshape(-1, 1)
            yv = yv - yv.mean()
            pls.fit(Z, yv)
        return pls.transform(Z).astype(float, copy=False)

    return Z
