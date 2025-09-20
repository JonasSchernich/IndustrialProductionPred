
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def make_global_lags(X: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    lags = sorted({int(l) for l in lags if int(l) >= 1})
    if not lags:
        return pd.DataFrame(index=X.index)
    parts = []
    for l in lags:
        df = X.shift(l)
        df.columns = [f"{c}_lag{l}" for c in X.columns]
        parts.append(df)
    return pd.concat(parts, axis=1)

def make_per_feature_lags_by_corr(
    Xtr: pd.DataFrame, ytr: pd.Series, candidates: Iterable[int], topk: int = 1
) -> Dict[str, Tuple[int, ...]]:
    candidates = [int(l) for l in candidates if int(l) >= 1]
    mapping = {}
    for col in Xtr.columns:
        scores = []
        for l in candidates:
            s = Xtr[col].shift(l)
            r = abs(s.corr(ytr))
            r = 0.0 if np.isnan(r) or np.isinf(r) else float(r)
            scores.append((l, r))
        best = tuple(l for l, _ in sorted(scores, key=lambda t: t[1], reverse=True)[:topk])
        mapping[col] = best if best else (min(candidates),)
    return mapping

def apply_per_feature_lags(X: pd.DataFrame, lag_map: Dict[str, Tuple[int, ...]]) -> pd.DataFrame:
    parts = []
    for col, lags in lag_map.items():
        for l in lags:
            parts.append(X[col].shift(l).rename(f"{col}_lag{l}"))
    if not parts:
        return pd.DataFrame(index=X.index)
    return pd.concat(parts, axis=1)

def make_rolling_means(X: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    wins = sorted({int(w) for w in windows if int(w) >= 2})
    if not wins:
        return pd.DataFrame(index=X.index)
    parts = []
    for w in wins:
        df = X.rolling(window=w, min_periods=w).mean().shift(1)
        df.columns = [f"{c}_rm{w}" for c in X.columns]
        parts.append(df)
    return pd.concat(parts, axis=1)

def make_ema(X: pd.DataFrame, spans: Iterable[int]) -> pd.DataFrame:
    spans = sorted({int(s) for s in spans if int(s) >= 2})
    if not spans:
        return pd.DataFrame(index=X.index)
    parts = []
    for s in spans:
        df = X.ewm(span=s, adjust=False).mean().shift(1)
        df.columns = [f"{c}_ema{s}" for c in X.columns]
        parts.append(df)
    return pd.concat(parts, axis=1)

def build_engineered_matrix(
    X: pd.DataFrame,
    base_features: list[str],
    fe_spec: dict
) -> pd.DataFrame:
    Xb = X[base_features]
    parts = []
    if "lag_map" in fe_spec and fe_spec["lag_map"]:
        parts.append(apply_per_feature_lags(Xb, fe_spec["lag_map"]))
    if "lags" in fe_spec and fe_spec["lags"]:
        parts.append(make_global_lags(Xb, fe_spec["lags"]))
    if "rm_windows" in fe_spec and fe_spec["rm_windows"]:
        parts.append(make_rolling_means(Xb, fe_spec["rm_windows"]))
    if "ema_spans" in fe_spec and fe_spec["ema_spans"]:
        parts.append(make_ema(Xb, fe_spec["ema_spans"]))

    if not parts:
        return pd.DataFrame(index=X.index)
    M = pd.concat(parts, axis=1)
    return M.dropna(axis=1, how="all")

def apply_pca_train_transform(
    M_train: pd.DataFrame, M_eval: pd.DataFrame,
    pca_n: Optional[int] = None, pca_var: Optional[float] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pca_n is None and pca_var is None:
        return M_train, M_eval
    if pca_n is not None and pca_n <= 0:
        return M_train, M_eval

    scaler = StandardScaler(with_mean=True, with_std=True)
    Ztr = scaler.fit_transform(M_train.values)
    Zev = scaler.transform(M_eval.values)

    if pca_var is not None:
        pca = PCA(n_components=pca_var, svd_solver="full")
    else:
        pca = PCA(n_components=pca_n)

    Ztr_p = pca.fit_transform(Ztr)
    Zev_p = pca.transform(Zev)

    cols = [f"PC{i+1}" for i in range(Ztr_p.shape[1])]
    Mtr_p = pd.DataFrame(Ztr_p, index=M_train.index, columns=cols)
    Mev_p = pd.DataFrame(Zev_p, index=M_eval.index, columns=cols)
    return Mtr_p, Mev_p
