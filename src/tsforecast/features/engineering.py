from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np

# === Grundfunktionen ===

def make_global_lags(X: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    lags = list(sorted(set(int(l) for l in lags if int(l) > 0)))
    out = {}
    for c in X.columns:
        for L in lags:
            out[f"{c}_lag{L}"] = X[c].shift(L)
    M = pd.DataFrame(out, index=X.index)
    return M

def make_per_feature_lags_by_corr(
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    candidates: Iterable[int],
    topk: int = 1
) -> Dict[str, List[int]]:
    # wähle pro Basisfeature die Lags mit größter |corr| (Train)
    cand = [int(l) for l in candidates if int(l) > 0]
    corr_map = {}
    for c in Xtr.columns:
        vals = {}
        for L in cand:
            s = Xtr[c].shift(L)
            vals[L] = abs(s.corr(ytr))
        best = sorted(vals.items(), key=lambda kv: (-(0.0 if np.isnan(kv[1]) else kv[1]), kv[0]))[: max(1, int(topk))]
        corr_map[c] = [b[0] for b in best]
    return corr_map

def apply_per_feature_lags(X: pd.DataFrame, lag_map: Dict[str, List[int]]) -> pd.DataFrame:
    out = {}
    for c, Ls in lag_map.items():
        if c not in X.columns:
            continue
        for L in sorted(set(Ls)):
            out[f"{c}_lag{L}"] = X[c].shift(L)
    return pd.DataFrame(out, index=X.index)

def make_rolling_means(X: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    ws = [int(w) for w in windows if int(w) > 0]
    out = {}
    for c in X.columns:
        for w in ws:
            out[f"{c}_rm{w}"] = X[c].rolling(window=w, min_periods=w).mean().shift(1)
    return pd.DataFrame(out, index=X.index)

def make_ema(X: pd.DataFrame, spans: Iterable[int]) -> pd.DataFrame:
    ss = [int(s) for s in spans if int(s) > 0]
    out = {}
    for c in X.columns:
        for s in ss:
            out[f"{c}_ema{s}"] = X[c].ewm(span=s, adjust=False, min_periods=s).mean().shift(1)
    return pd.DataFrame(out, index=X.index)

# === Externe Blöcke ===

def _load_parquet_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq  # fallback
        return pq.read_table(path).to_pandas()

def _shift_all_cols(M: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    return M.shift(k)

def _merge_external_blocks(X: pd.DataFrame, fe_spec: Dict) -> pd.DataFrame:
    # nur externe, strikt geshiftete Features; kein X beilegen
    M = pd.DataFrame(index=X.index)

    tsfresh_path = fe_spec.get("tsfresh_path")
    if tsfresh_path:
        T = _load_parquet_safe(tsfresh_path)
        T = T.reindex(X.index)
        T = _shift_all_cols(T, 1)
        M = M.join(T, how="left")

    fm_pred_path = fe_spec.get("fm_pred_path")
    if fm_pred_path:
        F = _load_parquet_safe(fm_pred_path)
        F = F.reindex(X.index)
        F = _shift_all_cols(F, 1)
        M = M.join(F, how="left")

    return _cast_float32(M)


def _cast_float32(M: pd.DataFrame) -> pd.DataFrame:
    # nur numerische Spalten auf float32
    num = M.select_dtypes(include=[np.number]).columns
    M[num] = M[num].astype(np.float32)
    return M

# === Orchestrierung Engineering ===

def build_engineered_matrix(
    X: pd.DataFrame,
    base_features: List[str],
    fe_spec: Dict
) -> pd.DataFrame:
    """
    Kombiniert Lags/Per-Feature-Lags, RM, EMA und externe Blöcke. Leakage-sicher (shift in RM/EMA/externe).
    """
    Xb = X[base_features].copy()
    blocks = []

    # Lags
    lag_map = fe_spec.get("lag_map")
    lags = fe_spec.get("lags")
    if lag_map is not None:
        blocks.append(apply_per_feature_lags(Xb, lag_map))
    elif lags is not None:
        blocks.append(make_global_lags(Xb, lags))

    # Rolling / EMA (immer auf Basisfeatures; optional auch auf Lags – hier konservativ)
    rm_windows = fe_spec.get("rm_windows") or ()
    if len(rm_windows) > 0:
        blocks.append(make_rolling_means(Xb, rm_windows))

    ema_spans = fe_spec.get("ema_spans") or ()
    if len(ema_spans) > 0:
        blocks.append(make_ema(Xb, ema_spans))

    # Merge Blöcke
    if blocks:
        M = pd.concat(blocks, axis=1)
    else:
        M = pd.DataFrame(index=X.index)  # leer

    # Externe Blöcke (shifted)
    if ("tsfresh_path" in fe_spec and fe_spec["tsfresh_path"]) or ("fm_pred_path" in fe_spec and fe_spec["fm_pred_path"]):
        M = M.join(_merge_external_blocks(X, fe_spec), how="left")

    # float32
    M = _cast_float32(M)

    return M

# === PCA ===

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca_train_transform(
    M_train: pd.DataFrame,
    M_eval: pd.DataFrame,
    pca_n: Optional[int],
    pca_var: Optional[float],
    pca_solver: str = "auto"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize -> PCA (fit on Train) -> Transform Train/Eval.
    Hinweis: Bei pca_var erzwingt sklearn intern 'full'. Für int pca_n kann 'randomized' sinnvoll sein.
    """
    if (pca_n is None) and (pca_var is None):
        return M_train.copy(), M_eval.copy()

    # numerische Spalten
    cols = list(M_train.columns)
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(M_train.values)
    Zev = scaler.transform(M_eval.values)

    if pca_var is not None:
        n_comp = float(pca_var)
        solver = "full"
    else:
        n_comp = int(pca_n)
        solver = pca_solver if pca_solver in {"auto", "full", "randomized"} else "auto"

    pca = PCA(n_components=n_comp, svd_solver=solver)
    Ztr_p = pca.fit_transform(Ztr)
    Zev_p = pca.transform(Zev)

    pc_cols = [f"PC{i+1}" for i in range(Ztr_p.shape[1])]
    Mtr_p = pd.DataFrame(Ztr_p, index=M_train.index, columns=pc_cols).astype(np.float32)
    Mev_p = pd.DataFrame(Zev_p, index=M_eval.index, columns=pc_cols).astype(np.float32)
    return Mtr_p, Mev_p
