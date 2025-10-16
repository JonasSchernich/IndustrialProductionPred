
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import lstsq

from .config import CorrelationSpec

# ==========================================================
# Utilities
# ==========================================================

def _seasonal_needed(param):
    pass


# --- replace in src/features.py ---

def _design_nuisance(y: pd.Series, seasonal: str, I_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train-only Nuisance-Design D_τ für τ in 0..I_t-1 mit Ziel y_{τ+1}.
    Saisonale Terme sind DEAKTIVIERT. Es wird nur (1, y_{τ-1}) genutzt.
    Gibt (D, taus) zurück, wobei taus genau die Indizes der gültigen Trainingspaare sind.
    """
    # gültige τ: y_{τ+1} muss existieren, y_{τ-1} muss existieren
    taus = np.arange(0, I_t)
    taus = taus[(taus + 1) < I_t]     # y_{τ+1} existiert
    taus = taus[(taus - 1) >= 0]      # y_{τ-1} existiert
    if len(taus) == 0:
        return np.empty((0, 2)), taus

    const = np.ones_like(taus, dtype=float)
    y_tau_1 = y.shift(1).iloc[taus].to_numpy(dtype=float)  # y_{τ-1}
    D = np.column_stack([const, y_tau_1])
    return D, taus



def _residualize_vec(vec: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    OLS residuals of vec on columns of D (adds no intercept; include it in D).
    If D has zero rows, returns empty array.
    """
    if D.shape[0] == 0:
        return np.array([])
    beta, *_ = lstsq(D, vec, rcond=None)
    return vec - D @ beta

def _ewma_weights(n: int, lam: float) -> np.ndarray:
    w = lam ** np.arange(n-1, -1, -1)
    w = w / w.sum()
    return w

def pw_corr(r_x: np.ndarray, r_y: np.ndarray, spec: CorrelationSpec) -> float:
    """
    Prewhitened correlation given residuals r_x, r_y.
    spec.mode = "expanding" → standard correlation (equal weights)
    spec.mode = "ewma" → EWMA correlation with optional finite window
    """
    n = min(len(r_x), len(r_y))
    if n == 0:
        return 0.0
    rx = np.asarray(r_x[-n:], dtype=float)
    ry = np.asarray(r_y[-n:], dtype=float)

    # Guard against NaNs
    m = ~(np.isnan(rx) | np.isnan(ry))
    if m.sum() < 2:
        return 0.0
    rx, ry = rx[m], ry[m]
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0

    mode = spec.get("mode", "expanding")
    if mode == "expanding":
        return float(np.corrcoef(rx, ry)[0, 1])
    else:
        lam = spec.get("lam", 0.98) or 0.98
        W = spec.get("window", None)
        if W is not None and W < len(rx):
            rx = rx[-W:]
            ry = ry[-W:]
        n = len(rx)
        if n < 2:
            return 0.0
        w = _ewma_weights(n, lam)
        rx_c = rx - np.sum(w * rx)
        ry_c = ry - np.sum(w * ry)
        num = np.sum(w * rx_c * ry_c)
        den = np.sqrt(np.sum(w * rx_c**2) * np.sum(w * ry_c**2))
        return float(num / den) if den > 0 else 0.0

# ==========================================================
# Public API: Feature engineering & Screening
# ==========================================================

@dataclass
class FEState:
    D: np.ndarray
    taus: np.ndarray
    lag_map: Dict[int, List[int]]   # per-feature top-k lag(s)
    scores: Dict[str, float]        # per-column screening scores

def select_lags_per_feature(
    X: pd.DataFrame,
    y: pd.Series,
    I_t: int,
    L: List[int],
    k: int,
    corr_spec: CorrelationSpec,
    seasonal_policy: str = "auto"
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int,int], float], np.ndarray, np.ndarray]:
    """
    Returns (lag_map, lag_scores, D, taus).
    lag_map[j] = list of selected lags for feature j (length <= k)
    lag_scores[(j, lag)] = abs prewhitened corr score
    """
    D, taus = _design_nuisance(y, seasonal_policy, I_t)
    if len(taus) == 0:
        return {}, {}, D, taus

    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)
    r_y = _residualize_vec(y_next, D)

    lag_map: Dict[int, List[int]] = {}
    lag_scores: Dict[Tuple[int,int], float] = {}

    taus_min = taus.min() if len(taus) else 0

    for j, col in enumerate(X.columns):
        scores_j = []
        x = X[col].to_numpy(dtype=float)
        # simple per-feature mean for imputation if needed (train-only)
        col_mean = np.nanmean(x[:I_t]) if np.isnan(x[:I_t]).any() else None

        for lag in L:
            # ensure availability of x_{tau-lag}
            valid = taus[taus - lag >= 0] if (taus_min - lag < 0) else taus
            if len(valid) == 0:
                continue
            x_l = x[valid - lag]
            # impute NaNs in x_l with train-only mean if necessary
            if col_mean is not None and np.isnan(x_l).any():
                x_l = np.where(np.isnan(x_l), col_mean, x_l)
            # align D rows to the chosen valid taus
            D_slice = D[D.shape[0]-len(valid):] if len(valid) != len(taus) else D
            r_x = _residualize_vec(x_l, D_slice)
            score = abs(pw_corr(r_x, r_y[-len(r_x):], corr_spec))
            scores_j.append((lag, score))
            lag_scores[(j, lag)] = score

        scores_j.sort(key=lambda z: z[1], reverse=True)
        lag_map[j] = [lag for lag, _ in scores_j[:k]] if scores_j else []

    return lag_map, lag_scores, D, taus

def apply_rm3(X: pd.DataFrame) -> pd.DataFrame:
    # causal RM3 (min_periods=1) – keine NaN-Erzeugung am Fensteranfang
    return X.rolling(3, min_periods=1).mean()

def build_engineered_matrix(
    X: pd.DataFrame, lag_map: Dict[int, List[int]]
) -> pd.DataFrame:
    cols = []
    for j, col in enumerate(X.columns):
        # include original
        cols.append(pd.Series(X[col].values, index=X.index, name=f"{col}__lag0"))
        for lag in lag_map.get(j, []):
            if lag > 0:
                cols.append(pd.Series(X[col].shift(lag).values, index=X.index, name=f"{col}__lag{lag}"))
    return pd.concat(cols, axis=1)

def screen_k1(
    X_eng: pd.DataFrame,
    y: pd.Series,
    I_t: int,
    corr_spec: CorrelationSpec,
    D: np.ndarray,
    taus: np.ndarray,
    k1_topk: int = 50,
    threshold: Optional[float] = None
) -> Tuple[List[str], Dict[str, float]]:
    """
    Prewhitened corr screening with train-only NaN guards.
    """
    if len(taus) == 0:
        return [], {}

    y_next = y.shift(-1).iloc[taus].to_numpy(dtype=float)
    r_y = _residualize_vec(y_next, D)

    scores: Dict[str, float] = {}
    for c in X_eng.columns:
        xvals = X_eng[c].iloc[taus].to_numpy(dtype=float)
        # train-only mean imputation for screening
        if np.isnan(xvals).any():
            m = np.nanmean(xvals)
            if np.isnan(m):  # all NaN fallback
                scores[c] = 0.0
                continue
            xvals = np.where(np.isnan(xvals), m, xvals)
        r_x = _residualize_vec(xvals, D)
        scores[c] = abs(pw_corr(r_x, r_y, corr_spec))

    if threshold is not None:
        keep = [c for c, s in scores.items() if s >= threshold]
    else:
        keep = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k1_topk]]
    return keep, scores

def redundancy_reduce_greedy(
    X_sel: pd.DataFrame,
    corr_spec: CorrelationSpec,
    D: np.ndarray,
    taus: np.ndarray,
    threshold: float
) -> List[str]:
    """
    Greedy redundancy control on residualized series with NaN-guards.
    """
    if len(taus) == 0 or X_sel.shape[1] == 0:
        return []

    keep: List[str] = []
    feats = list(X_sel.columns)

    # precompute residualized columns (with mean impute if needed)
    rcols: Dict[str, np.ndarray] = {}
    for c in feats:
        xv = X_sel[c].iloc[taus].to_numpy(dtype=float)
        if np.isnan(xv).any():
            m = np.nanmean(xv)
            if not np.isnan(m):
                xv = np.where(np.isnan(xv), m, xv)
        rcols[c] = _residualize_vec(xv, D)

    for c in feats:
        ok = True
        for k in keep:
            r = abs(pw_corr(rcols[c], rcols[k], corr_spec))
            if r >= threshold:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep

# ==========================================================
# Dimensionality Reduction (train-only safe: imputer + scaler + PCA/PLS)
# ==========================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
try:
    from sklearn.cross_decomposition import PLSRegression
    _HAS_PLS = True
except Exception:
    PLSRegression = None
    _HAS_PLS = False

@dataclass
class DRMap:
    imputer_stats: Optional[Dict[str, Any]]   # {"means": np.ndarray}
    scaler: Optional[StandardScaler]
    pca: Optional[PCA]
    pls: Optional[Any]
    method: str
    cols: List[str]                           # columns kept during DR fit
    pca_k: Optional[int] = None               # retained PCs if PCA

def _fit_imputer_train_only(X_tr: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute simple train-only column means for imputation.
    Also returns the list of columns actually used (drops all-NaN columns).
    """
    cols = list(X_tr.columns)
    # Drop all-NaN columns (cannot be imputed meaningfully)
    mask_all_nan = X_tr.isna().all(axis=0)
    if mask_all_nan.any():
        X_tr = X_tr.loc[:, ~mask_all_nan]
        cols = list(X_tr.columns)
    means = X_tr.mean(axis=0, skipna=True).to_numpy(dtype=float)
    return {"means": means, "cols": cols}

def _apply_imputer(X: pd.DataFrame, imp: Dict[str, Any]) -> pd.DataFrame:
    cols = imp["cols"]
    means = imp["means"]
    X2 = X.reindex(columns=cols)
    if X2.shape[1] == 0:
        return X2
    arr = X2.to_numpy(dtype=float)
    # per-column imputation
    for j in range(arr.shape[1]):
        col = arr[:, j]
        m = means[j]
        # If mean is NaN (shouldn't happen after all-NaN drop), fallback to zero
        if np.isnan(m):
            m = 0.0
        col[np.isnan(col)] = m
        arr[:, j] = col
    return pd.DataFrame(arr, index=X2.index, columns=cols)

def fit_dr(
    X_tr: pd.DataFrame,
    method: str = "none",
    pca_var_target: float = 0.95,
    pca_kmax: int = 25,
    pls_components: int = 2
) -> DRMap:
    """
    Train-only DR fit with NaN-safe imputation and column alignment.
    Order: impute(train) -> standardize(train) -> PCA/PLS fit.
    """
    # 1) Train-only imputer
    imp = _fit_imputer_train_only(X_tr)
    X_imp = _apply_imputer(X_tr, imp)

    # 2) Standardize
    if X_imp.shape[1] > 0:
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_imp.values)
        Xs = scaler.transform(X_imp.values)
    else:
        scaler = None
        Xs = X_imp.values

    if method == "pca":
        if Xs.shape[1] == 0:
            return DRMap(imp, scaler, None, None, "pca", imp["cols"], pca_k=0)
        # fit PCA with at most pca_kmax comps
        ncomp = int(min(pca_kmax, Xs.shape[1]))
        pca = PCA(n_components=ncomp, svd_solver="auto", random_state=0).fit(Xs)
        # enforce variance target by truncation if requested (<1.0)
        if pca_var_target < 1.0 and pca.explained_variance_ratio_.size > 0:
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            k = int(np.searchsorted(cumsum, pca_var_target) + 1)
            k = max(1, min(k, ncomp))
        else:
            k = ncomp
        return DRMap(imp, scaler, pca, None, "pca", imp["cols"], pca_k=k)

    elif method == "pls":
        if not _HAS_PLS:
            raise RuntimeError("PLS requested but sklearn.cross_decomposition not available.")
        c = int(max(1, min(pls_components, Xs.shape[1] if Xs.ndim == 2 else 0)))
        pls = PLSRegression(n_components=c, scale=True) if c > 0 else None
        # PLS wird (falls fit_pls=True) später mit y im Training gefittet
        return DRMap(imp, scaler, None, pls, "pls", imp["cols"])

    else:
        # pass-through (nur Imputer + Scaler)
        return DRMap(imp, scaler, None, None, "none", imp["cols"])

def transform_dr(
    mapper: DRMap,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    fit_pls: bool = False
) -> np.ndarray:
    """
    Apply DR map to any matrix (train or eval row): align cols -> impute(train means) -> scale -> project.
    For PLS: if fit_pls=True, fits PLS on provided (X,y) (training slice); otherwise uses stored pls if present.
    """
    # 1) Align to training columns and impute with train-only means
    X_imp = _apply_imputer(X, {"means": mapper.imputer_stats["means"], "cols": mapper.cols}) if mapper.imputer_stats else X.copy()

    # 2) Scale
    if mapper.scaler is not None and X_imp.shape[1] > 0:
        Xs = mapper.scaler.transform(X_imp.values)
    else:
        Xs = X_imp.values

    # 3) Project
    if mapper.method == "pca" and mapper.pca is not None:
        Z = mapper.pca.transform(Xs)
        k = mapper.pca_k or Z.shape[1]
        Z = Z[:, :k]
        return Z
    elif mapper.method == "pls" and mapper.pls is not None:
        if fit_pls:
            if y is None:
                raise ValueError("transform_dr(..., fit_pls=True) requires y for PLS.")
            mapper.pls.fit(Xs, y.values if hasattr(y, "values") else np.asarray(y))
            Z = mapper.pls.transform(Xs)
            return Z
        else:
            # If PLS not fitted (no train call), project if possible; else pass-through
            try:
                Z = mapper.pls.transform(Xs)
                return Z
            except Exception:
                return Xs
    else:
        return Xs

# ==========================================================
# Target-only placeholder blocks (can be overridden in notebook)
# ==========================================================

def tsfresh_block(y: pd.Series, I_t: int, W: int = 12) -> pd.DataFrame:
    # Simple interpretable stats over last W months
    start = max(0, I_t - W)
    window = y.iloc[start:I_t]
    feats = {
        "ts_mean": window.mean(),
        "ts_std": window.std(ddof=0),
        "ts_ac1": window.autocorr(lag=1) if window.size > 1 else 0.0,
    }
    return pd.DataFrame({k: [v] for k, v in feats.items()}, index=[y.index[I_t - 1]])

def chronos_block(y: pd.Series, I_t: int, W: int = 12) -> pd.DataFrame:
    # Placeholder: summarise last W values as "predictive distribution" moments
    start = max(0, I_t - W)
    window = y.iloc[start:I_t]
    feats = {
        "ch_mu": window.mean(),
        "ch_p50": window.median(),
        "ch_sigma": window.std(ddof=0),
        "ch_p10": window.quantile(0.1),
        "ch_p90": window.quantile(0.9),
    }
    return pd.DataFrame({k: [v] for k, v in feats.items()}, index=[y.index[I_t - 1]])
