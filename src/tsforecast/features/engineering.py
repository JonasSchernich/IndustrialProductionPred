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
    candidates,
    topk: int = 1
) -> dict:
    """
    Wählt je Basisfeature die Lags mit maximaler |corr(X.shift(L), y)|,
    berechnet auf dem Trainingsfenster in exakt der finalen Ausrichtung.
    Kein zusätzlicher shift(1) für Lags – der gilt nur für externe Blöcke.
    """
    # Kandidaten normalisieren
    if candidates is None:
        cand = [1, 2, 3, 6, 12]
    elif isinstance(candidates, (list, tuple)) and len(candidates) > 0 and isinstance(candidates[0], (list, tuple, set)):
        s = set()
        for block in candidates:
            s.update(int(l) for l in block)
        cand = sorted(x for x in s if x > 0)
    else:
        cand = sorted({int(l) for l in candidates if int(l) > 0})

    lag_map = {}
    yv = ytr.astype(float)

    for c in Xtr.columns:
        x = Xtr[c].astype(float)
        stats = []
        for L in cand:
            z = x.shift(L)
            ok = z.notna() & yv.notna()
            if ok.sum() < 10:
                continue
            r = np.corrcoef(z[ok].values, yv[ok].values)[0, 1]
            if np.isfinite(r):
                stats.append((abs(float(r)), L))
        stats.sort(key=lambda z: z[0], reverse=True)
        if not stats:
            continue
        chosen = [L for (_, L) in stats[:max(1, int(topk))]]
        lag_map[c] = chosen

    return lag_map


def apply_per_feature_lags(X: pd.DataFrame, lag_map: Dict[str, List[int]]) -> pd.DataFrame:
    out = {}
    for c, Ls in lag_map.items():
        if c not in X.columns:
            continue
        for L in sorted(set(Ls)):
            out[f"{c}_lag{L}"] = X[c].shift(L)
    return pd.DataFrame(out, index=X.index)

def make_rolling_means_shifted(X: pd.DataFrame, windows) -> pd.DataFrame:
    windows = sorted({int(w) for w in windows if int(w) > 1})
    out = {}
    Xs = X.shift(1)  # F_t: Zeile t nutzt nur <= t
    for c in X.columns:
        s = Xs[c]
        for w in windows:
            out[f"{c}_rm{w}_lag1"] = s.rolling(w, min_periods=w).mean()
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
    # extern gelieferte Blöcke; standardmäßig konservativ mit shift(1)
    ext_shift = int(fe_spec.get("external_shift", 0))  # 1 = F_{t-1}, 0 = F_t (nur bei echter Inline-Berechnung!)
    M = pd.DataFrame(index=X.index)

    def _apply_cadence(df: pd.DataFrame, cadence: int) -> pd.DataFrame:
        if cadence <= 1 or df.empty:
            return df
        pos = np.arange(len(df), dtype=int)
        anyrow = df.notna().any(axis=1).to_numpy()
        first = int(np.argmax(anyrow)) if anyrow.any() else 0
        mask = ((pos - first) % cadence) == 0
        return df.where(pd.Series(mask, index=df.index), np.nan)

    def _prep(df: pd.DataFrame, cadence: int, shift: int) -> pd.DataFrame:
        df = df.reindex(X.index)
        df = _apply_cadence(df, cadence)
        if shift != 0:
            df = df.shift(shift)
        if cadence > 1:
            df = df.ffill()
        return df

    # TSFresh
    tsfresh_path = fe_spec.get("tsfresh_path")
    if tsfresh_path:
        T = _load_parquet_safe(tsfresh_path)
        pref = fe_spec.get("tsfresh_prefix", "")
        if pref:
            T = T.add_prefix(pref)
        cad = int(fe_spec.get("tsfresh_cadence", 1) or 1)
        T = _prep(T, cadence=cad, shift=ext_shift)
        M = M.join(T, how="left")

    # Foundation/Chronos
    fm_pred_path = fe_spec.get("fm_pred_path")
    if fm_pred_path:
        F = _load_parquet_safe(fm_pred_path)
        pref = fe_spec.get("fm_prefix", "")
        if pref:
            F = F.add_prefix(pref)
        cad = int(fe_spec.get("fm_cadence", 1) or 1)
        F = _prep(F, cadence=cad, shift=ext_shift)
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
    Kombiniert Lags/Per-Feature-Lags, RM und externe Blöcke.
    Leakage-sicher: RM kausal ohne extra Shift; externe Blöcke mit shift(1).
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

    # Rolling Mean (immer auf Basisfeatures; konservativ)
    rm_windows = fe_spec.get("rm_windows") or ()
    if len(rm_windows) > 0:
        blocks.append(make_rolling_means_shifted(Xb, rm_windows))

    # no EMA

    # Merge Blöcke
    if blocks:
        M = pd.concat(blocks, axis=1)
    else:
        M = pd.DataFrame(index=X.index)

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
