from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional
import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- basic FE blocks ----

def make_global_lags(X: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    lags = sorted({int(l) for l in lags if int(l) >= 1})
    if not lags:
        return pd.DataFrame(index=X.index)
    parts = []
    for l in lags:
        df = X.shift(l)
        df.columns = [f"{c}_lag{l}" for c in X.columns]
        parts.append(df)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

def make_per_feature_lags_by_corr(Xtr: pd.DataFrame, ytr: pd.Series, candidates: Iterable[int], topk: int) -> Dict[str, Tuple[int, ...]]:
    lags = sorted({int(l) for l in candidates if int(l) >= 1})
    out = {}
    for c in Xtr.columns:
        tmp = {l: Xtr[c].shift(l).corr(ytr) for l in lags}
        # rank by |corr|
        best = sorted(tmp.items(), key=lambda kv: abs(kv[1] if kv[1] is not None else 0.0), reverse=True)[:topk]
        out[c] = tuple(l for l, _ in best)
    return out

def apply_per_feature_lags(X: pd.DataFrame, lag_map: Dict[str, Tuple[int, ...]]) -> pd.DataFrame:
    parts = []
    for c, lags in lag_map.items():
        for l in lags:
            parts.append(X[c].shift(l).rename(f"{c}_lag{l}"))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

def make_rolling_means(X: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    wins = sorted({int(w) for w in windows if int(w) >= 1})
    parts = []
    for w in wins:
        df = X.rolling(window=w, min_periods=w).mean().shift(1)
        df.columns = [f"{c}_rm{w}" for c in X.columns]
        parts.append(df)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

def make_ema(X: pd.DataFrame, spans: Iterable[int]) -> pd.DataFrame:
    spans = sorted({int(s) for s in spans if int(s) >= 1})
    parts = []
    for s in spans:
        df = X.ewm(span=s, adjust=False).mean().shift(1)
        df.columns = [f"{c}_ema{s}" for c in X.columns]
        parts.append(df)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

# ---- external blocks (tsfresh, Foundational Model stack) ----

_EXT_CACHE = {}

def _load_ext_features(path: str) -> pd.DataFrame:
    if path in _EXT_CACHE:
        return _EXT_CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    _EXT_CACHE[path] = df
    return df

def _maybe_merge_external_blocks(M: pd.DataFrame, X_index, fe_spec: dict) -> pd.DataFrame:
    # tsfresh precomputed features
    tsf_path = fe_spec.get("tsfresh_path") if isinstance(fe_spec, dict) else None
    if tsf_path:
        F = _load_ext_features(tsf_path)
        F = F.reindex(X_index).shift(1)  # safety
        M = pd.concat([M, F], axis=1)
    # Foundational model predictions as feature(s)
    fm_path = fe_spec.get("fm_pred_path") if isinstance(fe_spec, dict) else None
    if fm_path:
        S = _load_ext_features(fm_path)
        S = S.reindex(X_index).shift(1)
        M = pd.concat([M, S], axis=1)
    return M

# ---- master builders ----

def build_engineered_matrix(X: pd.DataFrame, base_features: Iterable[str], fe_spec: Dict) -> pd.DataFrame:
    parts = []
    if "lag_map" in fe_spec and fe_spec["lag_map"]:
        parts.append(apply_per_feature_lags(X[base_features], fe_spec["lag_map"]))
    else:
        parts.append(make_global_lags(X[base_features], fe_spec.get("lags", [])))

    parts.append(make_rolling_means(X[base_features], fe_spec.get("rm_windows", [])))
    parts.append(make_ema(X[base_features], fe_spec.get("ema_spans", [])))

    M = pd.concat([p for p in parts if p is not None and p.shape[1] > 0], axis=1) if parts else pd.DataFrame(index=X.index)
    # merge in external blocks if provided
    M = _maybe_merge_external_blocks(M, X.index, fe_spec)
    return M

def apply_pca_train_transform(M_train: pd.DataFrame, M_eval: pd.DataFrame, pca_n: Optional[int], pca_var: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize -> PCA (fit on Train) -> transform Train/Eval. If both None, passthrough."""
    if (pca_n is None) and (pca_var is None):
        return M_train.copy(), M_eval.copy()

    scaler = StandardScaler()
    Ztr = scaler.fit_transform(M_train.values)
    Zev = scaler.transform(M_eval.values)

    pca = PCA(n_components=pca_var, svd_solver="full") if (pca_var is not None) else PCA(n_components=pca_n)
    Ztr_p = pca.fit_transform(Ztr)
    Zev_p = pca.transform(Zev)

    cols = [f"PC{i+1}" for i in range(Ztr_p.shape[1])]
    Mtr_p = pd.DataFrame(Ztr_p, index=M_train.index, columns=cols)
    Mev_p = pd.DataFrame(Zev_p, index=M_eval.index, columns=cols)
    return Mtr_p, Mev_p
