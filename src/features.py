# src/features.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import lstsq

# --- kleine Helfer ---
def _spec_get(spec: Any, key: str, default=None):
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)

# ==========================================================
# Residualisierung & Korrelation
# ==========================================================
def _residualize_vec(vec: np.ndarray, D: np.ndarray) -> np.ndarray:
    if D is None or D.size == 0:
        return vec.astype(float, copy=False)
    if D.shape[0] != len(vec):
        raise ValueError(f"D and vec length mismatch: D={D.shape}, vec={len(vec)}")
    beta, *_ = lstsq(D, vec, rcond=None)
    return vec - D @ beta

def _ewma_weights(n: int, lam: float) -> np.ndarray:
    w = lam ** np.arange(n - 1, -1, -1)
    s = w.sum()
    return (w / s) if s != 0 else (np.ones(n) / n)

def _corr_equal_weight(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x @ x) * (y @ y))
    if den <= 0:
        return 0.0
    return float((x @ y) / den)

def _corr_ewma(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    w = _ewma_weights(n, lam)
    xm = (w * x).sum(); ym = (w * y).sum()
    xc = x - xm; yc = y - ym
    num = float((w * xc * yc).sum())
    den = np.sqrt(float((w * xc * xc).sum()) * float((w * yc * yc).sum()))
    if den <= 0:
        return 0.0
    return num / den

def pw_corr(r_x: np.ndarray, r_y: np.ndarray, spec: Any) -> float:
    n = min(len(r_x), len(r_y))
    if n <= 1:
        return 0.0
    r_x = r_x[-n:].astype(float, copy=False)
    r_y = r_y[-n:].astype(float, copy=False)
    mode = _spec_get(spec, "mode", "expanding")
    if mode == "ewm":
        lam = float(_spec_get(spec, "lambda", 0.98))
        return _corr_ewma(r_x, r_y, lam)
    return _corr_equal_weight(r_x, r_y)

# ==========================================================
# Lag-Selektion & Engineering
# ==========================================================
def _nuisance_matrix(y: pd.Series, taus: np.ndarray) -> np.ndarray:
    y_lag1 = y.shift(1).iloc[taus].to_numpy(dtype=float)
    y_lag1 = np.nan_to_num(y_lag1, nan=0.0, posinf=0.0, neginf=0.0)
    ones = np.ones((len(taus), 1), dtype=float)
    return np.concatenate([ones, y_lag1.reshape(-1, 1)], axis=1)

def select_lags_per_feature(
    X: pd.DataFrame,
    y: pd.Series,
    I_t: int,
    L: Tuple[int, ...],
    k: int,
    corr_spec: Any,
) -> Tuple[Dict[str, List[int]], Dict[str, float], np.ndarray, np.ndarray]:
    """
    Wählt je Feature die Top-k Lags aus L anhand |corr(x_{j,τ-ℓ}, y_{τ+1})|
    für τ = 0..I_t-1 (train-only).
    Rückgabe: (lag_map, scores_any, D, taus)
    """
    taus = np.arange(int(I_t), dtype=int)
    D = _nuisance_matrix(y, taus)

    # Ziel: y_{τ+1} resid
    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)
    r_y = _residualize_vec(y_next, D)

    lag_map: Dict[str, List[int]] = {}
    scores_any: Dict[str, float] = {}

    for col in X.columns:
        s = X[col].astype(float)
        lag_scores: List[Tuple[float, int]] = []
        for lag in L:
            x = s.shift(lag).iloc[taus].to_numpy(dtype=float)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            r_x = _residualize_vec(x, D)
            sc = abs(pw_corr(r_x, r_y, corr_spec))
            lag_scores.append((sc, lag))
        lag_scores.sort(key=lambda z: z[0], reverse=True)
        chosen = [lag for (_, lag) in lag_scores[:max(1, int(k))]]
        lag_map[col] = sorted(set(chosen))
        scores_any[col] = lag_scores[0][0] if lag_scores else 0.0

    return lag_map, scores_any, D, taus

def build_engineered_matrix(X: pd.DataFrame, lag_map: Dict[str, List[int]]) -> pd.DataFrame:
    out = {}
    for col, lags in lag_map.items():
        s = X[col]
        for lag in lags:
            out[f"{col}__lag{lag}"] = s.shift(lag).astype(float)
    return pd.DataFrame(out, index=X.index)

def apply_rm3(X_eng: pd.DataFrame) -> pd.DataFrame:
    return X_eng.rolling(window=3, min_periods=1).mean()

# ==========================================================
# Screening & Redundanz (train-only; prewhitened)
# ==========================================================
def _align_D_to_taus(D: np.ndarray, taus: np.ndarray) -> np.ndarray:
    n = len(taus)
    if D.shape[0] == n:
        return D
    if D.shape[0] > n:
        return D[-n:, :]
    raise ValueError(f"D has fewer rows ({D.shape[0]}) than taus ({n}).")

def screen_k1(
    X_eng: pd.DataFrame,
    y: pd.Series,
    I_t: int,
    corr_spec: Any,
    D: np.ndarray,
    taus: np.ndarray,
    k1_topk: Optional[int] = 200,
    threshold: Optional[float] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Score je Spalte: |corr(r_x, r_y)| auf 'taus' (prewhitened).
    Auswahl top-k oder per Schwelle.
    """
    if len(taus) == 0 or X_eng.shape[1] == 0:
        return [], {}

    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)
    D_slice = _align_D_to_taus(D, taus)
    r_y = _residualize_vec(y_next, D_slice)

    scores: Dict[str, float] = {}
    for col in X_eng.columns:
        xvals = X_eng[col].iloc[taus].to_numpy(dtype=float)
        xvals = np.nan_to_num(xvals, nan=0.0, posinf=0.0, neginf=0.0)
        r_x = _residualize_vec(xvals, D_slice)
        scores[col] = abs(pw_corr(r_x, r_y, corr_spec))

    cols = list(X_eng.columns)
    if threshold is not None:
        keep = [c for c in cols if scores.get(c, 0.0) >= float(threshold)]
    else:
        k = int(k1_topk or 0)
        keep = [] if k <= 0 else [c for c, _ in sorted(scores.items(), key=lambda z: z[1], reverse=True)[:k]]
    return keep, scores

def redundancy_reduce_greedy(
    X_sel: pd.DataFrame,
    corr_spec: Any,
    D: np.ndarray,
    taus: np.ndarray,
    redundancy_param: float = 0.90,
    scores: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Greedy: behalte Spalten in absteigender Score-Reihenfolge, solange
    |corr_prewhite(r_x_j, r_x_k)| <= redundancy_param für bereits behaltene.
    """
    if X_sel.shape[1] <= 1:
        return list(X_sel.columns)

    D_slice = _align_D_to_taus(D, taus)

    order = list(X_sel.columns)
    if scores:
        order.sort(key=lambda c: scores.get(c, 0.0), reverse=True)
    else:
        order.sort()

    # Residuen vorrechnen
    R: Dict[str, np.ndarray] = {}
    for c in order:
        xv = X_sel[c].iloc[taus].to_numpy(dtype=float)
        xv = np.nan_to_num(xv, nan=0.0, posinf=0.0, neginf=0.0)
        R[c] = _residualize_vec(xv, D_slice)

    kept: List[str] = []
    for c in order:
        ok = True
        for kcol in kept:
            if abs(pw_corr(R[c], R[kcol], corr_spec)) > float(redundancy_param):
                ok = False
                break
        if ok:
            kept.append(c)
    return kept

# ==========================================================
# DR (Standardisierung, PCA/PLS)
# ==========================================================
@dataclass
class _DRMap:
    method: str
    cols_: List[str]
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None
    pca_components_: Optional[np.ndarray] = None
    pls_model: Optional[object] = None  # sklearn PLSRegression

def _fit_standardizer(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X.to_numpy(dtype=float), axis=0)
    sd = np.nanstd(X.to_numpy(dtype=float), axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return mu.astype(float), sd.astype(float)

def _apply_standardizer(X: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X.to_numpy(dtype=float) - mu.reshape(1, -1)) / sd.reshape(1, -1)

def fit_dr(
    X_tr: pd.DataFrame,
    method: str = "none",
    pca_var_target: float = 0.95,
    pca_kmax: int = 50,
    pls_components: int = 4,
) -> _DRMap:
    """
    Train-only Fit der DR. Für PCA/PLS vorher standardisieren (train-Stats),
    danach Eingaben immer auf finite Werte clippen.
    """
    m = _DRMap(method=method, cols_=list(X_tr.columns))

    if method == "none":
        return m

    mu, sd = _fit_standardizer(X_tr)
    m.scaler_mean, m.scaler_std = mu, sd

    if method == "pca":
        from sklearn.decomposition import PCA
        Z = _apply_standardizer(X_tr, mu, sd)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)   # <-- fix
        pca = PCA(svd_solver="full")
        pca.fit(Z)
        evr = pca.explained_variance_ratio_
        k = int(np.searchsorted(np.cumsum(evr), pca_var_target) + 1)
        k = int(min(max(1, k), pca_kmax, Z.shape[1]))
        m.pca_components_ = pca.components_[:k, :].copy()
        return m

    if method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        Z = _apply_standardizer(X_tr, mu, sd)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)   # <-- fix
        pls = PLSRegression(n_components=int(pls_components), scale=False)
        m.pls_model = pls
        return m

    m.method = "none"
    return m

def transform_dr(
    m: _DRMap,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    fit_pls: bool = False
) -> np.ndarray:
    if m.method == "none" or (m.scaler_mean is None or m.scaler_std is None):
        return X.to_numpy(dtype=float)

    # Spaltenauswahl konsistent halten
    Xc = X.loc[:, m.cols_]
    Z = _apply_standardizer(Xc, m.scaler_mean, m.scaler_std)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)         # <-- fix

    if m.method == "pca":
        V = m.pca_components_
        if V is None:
            return Z
        return (Z @ V.T).astype(float, copy=False)

    if m.method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        pls: PLSRegression = m.pls_model
        if fit_pls:
            if y is None:
                raise ValueError("PLS transform with fit_pls=True requires y.")
            yv = y.to_numpy(dtype=float).reshape(-1, 1)
            yv = yv - yv.mean()
            pls.fit(Z, yv)
        T_scores = pls.transform(Z)
        return T_scores.astype(float, copy=False)

    return Z
