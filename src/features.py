# src/features.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import lstsq

# Falls CorrelationSpec als Dict übergeben wird, robust darauf reagieren
def _spec_get(spec: Any, key: str, default=None):
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)

# ==========================================================
# Residualisierung & Korrelation
# ==========================================================

def _residualize_vec(vec: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Residualisiere 'vec' auf D per OLS, train-only.
    Erwartet: D.shape[0] == len(vec).
    """
    if D is None or D.size == 0:
        return vec.astype(float, copy=False)
    if D.shape[0] != len(vec):
        raise ValueError(f"D and vec length mismatch: D={D.shape}, vec={len(vec)}")
    beta, *_ = lstsq(D, vec, rcond=None)
    return vec - D @ beta

def _ewma_weights(n: int, lam: float) -> np.ndarray:
    w = lam ** np.arange(n - 1, -1, -1)
    s = w.sum()
    if s == 0:
        return np.ones(n) / n
    return w / s

def _corr_equal_weight(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    if denom <= 0:
        return 0.0
    return float((x @ y) / denom)

def _corr_ewma(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    w = _ewma_weights(n, lam)
    xm = (w * x).sum()
    ym = (w * y).sum()
    xc = x - xm
    yc = y - ym
    num = float((w * xc * yc).sum())
    den = np.sqrt(float((w * xc * xc).sum()) * float((w * yc * yc).sum()))
    if den <= 0:
        return 0.0
    return num / den

def pw_corr(r_x: np.ndarray, r_y: np.ndarray, spec: Any) -> float:
    """
    Prewhitened correlation zwischen Residuen r_x, r_y.
    spec.mode in {"expanding", "ewm"}; bei ewm optional 'lambda'.
    """
    n = min(len(r_x), len(r_y))
    if n <= 1:
        return 0.0
    r_x = r_x[-n:].astype(float, copy=False)
    r_y = r_y[-n:].astype(float, copy=False)
    mode = _spec_get(spec, "mode", "expanding")
    if mode == "ewm":
        lam = float(_spec_get(spec, "lambda", 0.98))
        return _corr_ewma(r_x, r_y, lam)
    # expanding (equal weights)
    return _corr_equal_weight(r_x, r_y)

# ==========================================================
# Lag-Selektion & Engineering
# ==========================================================

def _nuisance_matrix(y: pd.Series, taus: np.ndarray) -> np.ndarray:
    """
    D_tau = (1, y_{tau-1}) für tau in 'taus'. (Intercept + kurzer AR-Bezug)
    Achtung: y_{tau-1} kann am Kopf NaN sein; dort wird 0 eingesetzt.
    """
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
    Wählt je Feature die Top-k Lags aus L anhand |corr(x_{j,τ-ℓ}, y_{τ+1})| auf τ=0..I_t-1.
    Rückgabe:
      - lag_map: {col: [lag1, lag2, ...]}
      - scores_any: optional (hier: max-Score je Feature)
      - D: Nuisance-Matrix für τ=0..I_t-1
      - taus: np.arange(I_t)
    """
    taus = np.arange(int(I_t), dtype=int)  # 0..t_origin
    D = _nuisance_matrix(y, taus)

    lag_map: Dict[str, List[int]] = {}
    scores_any: Dict[str, float] = {}

    # Zielreihe (y_{τ+1}) für Trainingstau
    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)
    r_y = _residualize_vec(y_next, D)

    for col in X.columns:
        # Scores je Lag sammeln
        lag_scores: List[Tuple[float, int]] = []
        x_full = X[col].to_numpy(dtype=float)

        for lag in L:
            # x_{j,τ-ℓ} über die Indizes τ: shift nach hinten
            x_shifted = pd.Series(x_full).shift(lag).iloc[taus].to_numpy(dtype=float)
            x_shifted = np.nan_to_num(x_shifted, nan=0.0, posinf=0.0, neginf=0.0)
            r_x = _residualize_vec(x_shifted, D)
            sc = abs(pw_corr(r_x, r_y, corr_spec))
            lag_scores.append((sc, lag))

        # Top-k Lags wählen
        lag_scores.sort(key=lambda z: z[0], reverse=True)
        chosen = [lag for (_, lag) in lag_scores[:max(1, int(k))]]
        lag_map[col] = sorted(set(chosen))
        scores_any[col] = lag_scores[0][0] if lag_scores else 0.0

    return lag_map, scores_any, D, taus

def build_engineered_matrix(X: pd.DataFrame, lag_map: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Erzeugt Spalten {col__lagℓ} für alle in lag_map gewählten Lags.
    (Nur gelaggte Spalten; Head-Trim erfolgt später.)
    """
    out = {}
    for col, lags in lag_map.items():
        s = X[col]
        for lag in lags:
            out[f"{col}__lag{lag}"] = s.shift(lag).astype(float)
    X_eng = pd.DataFrame(out, index=X.index)
    return X_eng

def apply_rm3(X_eng: pd.DataFrame) -> pd.DataFrame:
    """
    Kausale RM3-Glättung auf allen Spalten (Fenster=3, min_periods=1).
    Hinweis: Head-Trim in der Pipeline berücksichtigt den RM3-Offset (2).
    """
    return X_eng.rolling(window=3, min_periods=1).mean()

# ==========================================================
# Screening & Redundanz (train-only; prewhitened)
# ==========================================================

def _align_D_to_taus(D: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Align D auf genau len(taus) Zeilen. In der Pipeline ist 'taus' stets eine
    Suffixmenge (nach Head-Trim). Daher genügt 'von unten' zu schneiden.
    """
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
    Prewhitened Screening:
      1) y_next = y_{τ+1} auf 'taus'
      2) D auf 'taus' alignen
      3) je Feature: r_x, r_y residualisieren; Score = |corr(r_x, r_y)|
      4) Top-K nach Score ODER Schwellenfilter
    Rückgabe: (Liste behaltene Spalten, Mapping col->Score)
    """
    if len(taus) == 0 or X_eng.shape[1] == 0:
        return [], {}

    # 1) y_{τ+1}
    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)

    # 2) D konsequent slicen
    D_slice = _align_D_to_taus(D, taus)

    # 3) Ziel-Residual
    r_y = _residualize_vec(y_next, D_slice)

    scores: Dict[str, float] = {}
    for col in X_eng.columns:
        xvals = X_eng[col].iloc[taus].to_numpy(dtype=float)
        xvals = np.nan_to_num(xvals, nan=0.0, posinf=0.0, neginf=0.0)
        r_x = _residualize_vec(xvals, D_slice)
        sc = abs(pw_corr(r_x, r_y, corr_spec))
        scores[col] = sc

    # 4) Auswahl
    cols = list(X_eng.columns)
    if threshold is not None:
        keep = [c for c in cols if scores.get(c, 0.0) >= float(threshold)]
    else:
        k = int(k1_topk or 0)
        if k <= 0:
            keep = []
        else:
            keep = [c for c, _ in sorted(scores.items(), key=lambda z: z[1], reverse=True)[:k]]
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
    Greedy-Redundanzfilter: behalte Spalten in absteigender Score-Reihenfolge,
    solange |corr_prewhite(r_x_j, r_x_k)| <= redundancy_param für alle bereits
    behaltenen.
    """
    if X_sel.shape[1] <= 1:
        return list(X_sel.columns)

    # D-Align sicherstellen
    D_slice = _align_D_to_taus(D, taus)

    # Reihenfolge nach Score (oder Alphabet)
    order = list(X_sel.columns)
    if scores:
        order.sort(key=lambda c: scores.get(c, 0.0), reverse=True)
    else:
        order.sort()

    # Precompute Residuen aller Kandidaten auf 'taus'
    R: Dict[str, np.ndarray] = {}
    for c in order:
        xv = X_sel[c].iloc[taus].to_numpy(dtype=float)
        xv = np.nan_to_num(xv, nan=0.0, posinf=0.0, neginf=0.0)
        R[c] = _residualize_vec(xv, D_slice)

    kept: List[str] = []
    for c in order:
        r_c = R[c]
        ok = True
        for kcol in kept:
            r_k = R[kcol]
            dep = abs(pw_corr(r_c, r_k, corr_spec))
            if dep >= redundancy_param:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept

# ==========================================================
# Dimension Reduction (train-only)
# ==========================================================

@dataclass
class _DRMap:
    method: str = "none"
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None
    pca_components_: Optional[np.ndarray] = None
    pls_model: Any = None
    cols_: Optional[List[str]] = None

def _fit_standardizer(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0).to_numpy(dtype=float)
    sd = X.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    sd[sd == 0.0] = 1.0
    return mu, sd

def _apply_standardizer(X: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X.to_numpy(dtype=float) - mu.reshape(1, -1)) / sd.reshape(1, -1)
    return Z

def fit_dr(
    X_tr: pd.DataFrame,
    method: str = "none",
    pca_var_target: float = 0.95,
    pca_kmax: int = 50,
    pls_components: int = 4,
) -> _DRMap:
    """
    Fit train-only DR-Objekt. Für 'pca' wird vorher standardisiert.
    Für 'pls' wird ebenfalls standardisiert und anschließend PLS1 gefittet.
    """
    m = _DRMap(method=method, cols_=list(X_tr.columns))

    if method == "none":
        return m

    mu, sd = _fit_standardizer(X_tr)
    m.scaler_mean, m.scaler_std = mu, sd

    if method == "pca":
        from sklearn.decomposition import PCA
        Z = _apply_standardizer(X_tr, mu, sd)
        pca = PCA(svd_solver="full")
        pca.fit(Z)
        # Anzahl Komponenten nach Varianz-Ziel + Kappung
        evr = pca.explained_variance_ratio_
        k = int(np.searchsorted(np.cumsum(evr), pca_var_target) + 1)
        k = int(min(max(1, k), pca_kmax, Z.shape[1]))
        m.pca_components_ = pca.components_[:k, :].copy()
        return m

    if method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        Z = _apply_standardizer(X_tr, mu, sd)
        # PLS1 auf zentriertem y wird beim Transform berücksichtigt
        pls = PLSRegression(n_components=int(pls_components), scale=False)
        # Rückgabe speichert Modell; fit erfolgt in transform (mit y)
        m.pls_model = pls
        return m

    # Fallback: keine DR
    m.method = "none"
    return m

def transform_dr(
    m: _DRMap,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    fit_pls: bool = False
) -> np.ndarray:
    """
    Wendet DR-Map auf X an (eval: fit_pls=False).
    Für PLS: beim Train-Aufruf (fit_pls=True) mit y fitten, danach für Eval nur transformieren.
    """
    if m is None or m.method == "none":
        return X.to_numpy(dtype=float)

    # Spaltenauswahl konsistent halten
    Xc = X.loc[:, m.cols_]

    Z = _apply_standardizer(Xc, m.scaler_mean, m.scaler_std)

    if m.method == "pca":
        # Projektion: V_{1:K}^T * Z^T → (n,K)
        V = m.pca_components_  # shape (K, p)
        if V is None:
            return Z  # falls PCA nicht verfügbar, roh standardisiert zurückgeben
        return (Z @ V.T).astype(float, copy=False)

    if m.method == "pls":
        if m.pls_model is None:
            return Z
        from sklearn.cross_decomposition import PLSRegression
        pls: PLSRegression = m.pls_model
        if fit_pls:
            if y is None:
                raise ValueError("PLS transform with fit_pls=True requires y.")
            yv = y.to_numpy(dtype=float).reshape(-1, 1)
            # y zentrieren (Standard in PLS); Skalen sind bereits in Z
            yv = yv - yv.mean()
            pls.fit(Z, yv)
        T_scores = pls.transform(Z)  # (n, c)
        return T_scores.astype(float, copy=False)

    # Fallback
    return Z
