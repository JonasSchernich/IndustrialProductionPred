# src/features.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import lstsq


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _spec_get(spec: Any, key: str, default=None):
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _nuisance_matrix(y: pd.Series, taus: np.ndarray) -> np.ndarray:
    """Erstellt die Nuisance-Matrix [1, y_lag1] für gegebene Zeitstempel."""
    if taus.size == 0:
        return np.empty((0, 2), dtype=float)
    y_lag1 = y.shift(1).iloc[taus].to_numpy(dtype=float)
    # y_lag1 kann NaNs am Anfang haben (wenn taus=0 oder 1 startet)
    # Ersetze NaNs in y_lag1 (Nuisance) mit 0.0
    y_lag1 = np.nan_to_num(y_lag1, nan=0.0, posinf=0.0, neginf=0.0)
    ones = np.ones((len(taus), 1), dtype=float)
    return np.concatenate([ones, y_lag1.reshape(-1, 1)], axis=1)


def _residualize_vec(v: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Residualisiere Vektor v auf Nuisance-Matrix D via OLS."""
    if len(v) == 0 or D.size == 0:
        return np.asarray(v, dtype=float)
    # lstsq ist robust bei (nahezu) singulären D
    beta, *_ = lstsq(D, v, rcond=None)
    return v - D @ beta


def _corr_equal_weight(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = (np.linalg.norm(x) * np.linalg.norm(y))
    if den == 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


# KORRIGIERT: Stabilere "Lehrbuch"-Variante (Kritik 3, Bild 2)
def _corr_ewma(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    """EWMA-gewichtete Korrelation (stabile Ein-Pass-Variante)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0

    # Nimm die letzten n Punkte
    x = x[-n:]
    y = y[-n:]

    mx = 0.0
    my = 0.0
    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    alpha = 1.0 - lam  # (1-lam)

    # Starte mit dem ersten Wert als initialem Mittelwert
    mx = x[0]
    my = y[0]

    for i in range(1, n):
        xi = x[i]
        yi = y[i]

        dx = xi - mx
        dy = yi - my

        mx += alpha * dx
        my += alpha * dy

        cxx = lam * cxx + alpha * dx * (xi - mx)
        cyy = lam * cyy + alpha * dy * (yi - my)
        cxy = lam * cxy + alpha * dx * (yi - my)  # Welford's cross-product update

    den = np.sqrt(cxx * cyy)
    return 0.0 if den <= 0.0 else float(cxy / den)


# --------------------------------------------------------------------------------------
# FIXED: pw_corr mit optionalem Window + robustem EWMA-Handling
# --------------------------------------------------------------------------------------

def pw_corr(r_x: np.ndarray, r_y: np.ndarray, spec: Any) -> float:
    """Pairwise correlation wrapper with optional windowing and EWMA.
    Supports spec keys: mode in {"expanding","ewma","ewm"}, lambda/lam in (0,1),
    and optional integer 'window' to restrict to the most recent W points.
    """
    # Coerce to 1D float arrays
    r_x = np.asarray(r_x, dtype=float).ravel()
    r_y = np.asarray(r_y, dtype=float).ravel()
    n = min(len(r_x), len(r_y))
    if n <= 1:
        return 0.0

    # --- Windowing (optional) ---
    W = _spec_get(spec, "window", default=None)
    try:
        W = int(W) if W is not None else 0
    except Exception:
        W = 0
    if W > 0 and n > W:
        r_x = r_x[-W:]
        r_y = r_y[-W:]
        n = W

    mode = _spec_get(spec, "mode", "expanding")
    if mode == "ewma" or mode == "ewm":
        lam_key1 = _spec_get(spec, "lam")
        lam_key2 = _spec_get(spec, "lambda", 0.98)
        # KORRIGIERT: Clamping (Kritik 3, Bild 2)
        lam = float(lam_key1 if lam_key1 is not None else lam_key2)
        lam = max(1e-6, min(1.0 - 1e-6, lam))  # Clamp lam in (0, 1)

        return _corr_ewma(r_x, r_y, lam)

    return _corr_equal_weight(r_x, r_y)


# --------------------------------------------------------------------------------------
# FIXED: select_lags_per_feature – saubere Masken & gemeinsame Basis
# --------------------------------------------------------------------------------------

def select_lags_per_feature(
        X: pd.DataFrame,
        y: pd.Series,
        I_t: int,
        L: Tuple[int, ...],
        k: int,
        corr_spec: Any,
) -> Tuple[Dict[str, List[int]], Dict[str, float], np.ndarray, np.ndarray]:
    """Select per-feature top-k lags by |corr(r_x, r_y)| on a valid common basis.

    Fixes:
    - Use masks/slicing instead of zero-filling for x (avoids bias).
    - Residualize r_x and r_y on *exactly* the same nuisance basis D for each lag.
    - Cache r_y and D per lag for performance.
    - Ensure timebase starts at 1 to keep y.shift(1) valid in nuisance.

    Returns:
        lag_map: {feature -> sorted list of chosen lags}
        scores_any: {feature -> best score across lags}
        D_for_k1: nuisance matrix for lag=0 features (screen_k1)
        taus_for_k1: corresponding time indices
    """
    taus_base = np.arange(1, int(I_t), dtype=int)  # start at 1 due to y.shift(1)
    if taus_base.size < 2:  # Brauchen mind. 2 Datenpunkte
        return {}, {}, np.empty((0, 2), dtype=float), taus_base

    y_next_base = y.shift(-1).iloc[taus_base].to_numpy(dtype=float)
    valid_y_mask = ~np.isnan(y_next_base)
    taus_base = taus_base[valid_y_mask]
    y_next_base = y_next_base[valid_y_mask]
    if taus_base.size < 2:
        return {}, {}, _nuisance_matrix(y, taus_base), taus_base

    lag_artifacts: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    min_lag = min(L) if L else 0

    # KORRIGIERT: Head-Trim Off-by-One (Kritik 1)
    # Hier taus_base >= min_lag statt >
    taus_for_k1_mask = (taus_base >= min_lag)
    if np.sum(taus_for_k1_mask) == 0:
        taus_for_k1 = taus_base[-1:].copy()
    else:
        taus_for_k1 = taus_base[taus_for_k1_mask]

    D_for_k1 = _nuisance_matrix(y, taus_for_k1)

    for lag in set(L + (0,)):  # include 0 for screen_k1
        # KORRIGIERT: Head-Trim Off-by-One (Kritik 1)
        lag_mask = (taus_base >= lag)
        if np.sum(lag_mask) < 2:
            continue
        taus_eff = taus_base[lag_mask]
        y_next_eff = y_next_base[lag_mask]
        D_eff = _nuisance_matrix(y, taus_eff)
        r_y_eff = _residualize_vec(y_next_eff, D_eff)
        lag_artifacts[lag] = (r_y_eff, D_eff, taus_eff)

    lag_map: Dict[str, List[int]] = {}
    scores_any: Dict[str, float] = {}

    for col in X.columns:
        s = X[col].astype(float)
        lag_scores: List[Tuple[float, int]] = []
        for lag in L:
            arts = lag_artifacts.get(lag)
            if arts is None:
                lag_scores.append((0.0, lag))
                continue
            r_y_eff, D_eff, taus_eff = arts

            # Shift once, slice exakt auf gültige taus, dann maskieren
            x = s.shift(lag).iloc[taus_eff].to_numpy(dtype=float)
            x_mask = ~np.isnan(x)
            if np.sum(x_mask) < 2:
                lag_scores.append((0.0, lag))
                continue

            x_eff = x[x_mask]
            r_y_eff_masked = r_y_eff[x_mask]
            D_eff_masked = D_eff[x_mask, :]

            r_x_eff = _residualize_vec(x_eff, D_eff_masked)
            sc = abs(pw_corr(r_x_eff, r_y_eff_masked, corr_spec))
            lag_scores.append((sc, lag))

        lag_scores.sort(key=lambda z: z[0], reverse=True)
        chosen = [lag for (_, lag) in lag_scores[:max(1, int(k))]]
        lag_map[col] = sorted(set(chosen))
        scores_any[col] = lag_scores[0][0] if lag_scores else 0.0

    if 0 in lag_artifacts:
        _, D_for_k1, taus_for_k1 = lag_artifacts[0]

    return lag_map, scores_any, D_for_k1, taus_for_k1


# --------------------------------------------------------------------------------------
# Restliche Funktionen
# --------------------------------------------------------------------------------------

def build_engineered_matrix(X: pd.DataFrame, lag_map: Dict[str, List[int]]) -> pd.DataFrame:
    out = {}
    for col, lags in lag_map.items():
        s = X[col]
        for lag in lags:
            out[f"{col}__lag{lag}"] = s.shift(lag).astype(float)
    return pd.DataFrame(out, index=X.index)


def apply_rm3(X_eng: pd.DataFrame) -> pd.DataFrame:
    return X_eng.rolling(window=3, min_periods=1).mean()


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
    n_components_: int
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

        # Clamping für PCA (n_components darf nicht > n_samples oder n_features sein)
        k_max_effective = min(n_samples, n_features, int(pca_kmax))

        pca = PCA(svd_solver="full")
        pca.fit(Z)
        evr = pca.explained_variance_ratio_
        k = int(np.searchsorted(np.cumsum(evr), pca_var_target) + 1)
        k = int(min(max(1, k), k_max_effective))  # Clamp auf max

        m.pca_components_ = pca.components_[:k, :].copy()
        m.n_components_ = k  # Speichere die tatsächliche Komponentenanzahl
        return m

    if method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        Z = _apply_standardizer(X_tr, mu, sd)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        # KORRIGIERT: PLS Komponenten clampen (Kritik 4, Bild 3)
        n, p = Z.shape
        c_eff = int(np.clip(pls_components, 1, max(1, min(p, n - 1))))  # min auf n-1 (oder p)

        pls = PLSRegression(n_components=c_eff, scale=False)
        m.pls_model = pls
        m.n_components_ = c_eff  # Speichere die tatsächliche Komponentenanzahl
        return m

    m.method = "none"
    m.n_components_ = n_features
    return m


def transform_dr(
        m: _DRMap,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit_pls: bool = False
) -> np.ndarray:
    if m.method == "none":
        # Sicherstellen, dass die Spaltenauswahl auch bei 'none' greift
        Xc = X.loc[:, m.cols_] if m.cols_ else X
        return np.nan_to_num(Xc.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    if (m.scaler_mean is None or m.scaler_std is None):
        raise ValueError("DR map (PCA/PLS) ist nicht korrekt initialisiert (scaler fehlt).")

    # Spaltenauswahl konsistent halten
    Xc = X.loc[:, m.cols_]
    Z = _apply_standardizer(Xc, m.scaler_mean, m.scaler_std)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

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
