from typing import List, Optional
import numpy as np
import pandas as pd
from ..types import FeatureSelectCfg

# variance
def _variance_filter(X: pd.DataFrame, thresh: float) -> pd.DataFrame:
    v = X.var(axis=0)
    keep = v > float(thresh) if thresh > 0 else v > 0.0
    return X.loc[:, keep]

# month dummies
def _month_dummies(idx: pd.DatetimeIndex) -> pd.DataFrame:
    if not isinstance(idx, pd.DatetimeIndex):
        return pd.DataFrame(index=idx)
    m = idx.month.astype(int)
    D = pd.get_dummies(m, prefix="m", drop_first=True)
    D.index = idx
    return D

# residualize X and y on Z
def _residualize(M: pd.DataFrame, y: pd.Series, use_month_dummies: bool, use_y_lags: bool):
    Z = []
    if use_month_dummies:
        Z.append(_month_dummies(y.index))
    if use_y_lags:
        Z.append(pd.DataFrame({"yl1": y.shift(1).values, "yl12": y.shift(12).values}, index=y.index))
    Z = [z for z in Z if z is not None and z.shape[1] > 0]
    if len(Z) == 0:
        return M, y
    Z = pd.concat(Z, axis=1).astype(float)
    A = pd.concat([M, y, Z], axis=1).dropna()
    if A.empty:
        return M.iloc[0:0, :], y.iloc[0:0]
    cols_M = list(M.columns)
    X = A[cols_M].astype(float).values
    z = np.c_[np.ones((len(A),1)), A[Z.columns].astype(float).values]
    yy = A[y.name].astype(float).values
    bz, *_ = np.linalg.lstsq(z, yy, rcond=None)
    ry = yy - z @ bz
    Bz, *_ = np.linalg.lstsq(z, X, rcond=None)
    RX = X - z @ Bz
    Ry = pd.Series(ry, index=A.index, name=y.name)
    RX = pd.DataFrame(RX, index=A.index, columns=cols_M)
    return RX, Ry

# abs corr
def _abs_corr_with_y(M: pd.DataFrame, y: pd.Series) -> pd.Series:
    c = M.corrwith(y).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return c.sort_values(ascending=False)

# greedy redundancy
def _redundancy_filter(M: pd.DataFrame, order: List[str], tau: float) -> List[str]:
    if tau is None or tau <= 0:
        return order
    kept = []
    for c in order:
        if not kept:
            kept.append(c); continue
        r = np.max(np.abs(M[kept].corrwith(M[c]).values))
        if not np.isfinite(r) or r <= tau:
            kept.append(c)
    return kept

# API
def select_engineered_features(Mtr: pd.DataFrame, ytr: pd.Series, cfg: FeatureSelectCfg) -> List[str]:
    M = _variance_filter(Mtr, cfg.variance_thresh)

    if cfg.mode == "none":
        cols = list(M.columns)
    elif cfg.mode == "manual":
        cols = [c for c in (cfg.manual_cols or []) if c in M.columns]
    else:
        if getattr(cfg, "prewhiten", False):
            RX, Ry = _residualize(M, ytr, bool(getattr(cfg, "use_month_dummies", True)),
                                  bool(getattr(cfg, "use_y_lags", True)))
            M0, y0 = RX, Ry
        else:
            M0, y0 = M, ytr

        corr = _abs_corr_with_y(M0, y0)
        if cfg.mode in {"auto_topk_prewhite", "auto_topk"}:
            k = max(1, int(cfg.topk))
            order = corr.index.tolist()[:k]
        elif cfg.mode in {"auto_threshold_prewhite", "auto_threshold"}:
            thr = float(cfg.min_abs_corr or 0.0)
            order = corr[corr >= thr].index.tolist()
        else:
            raise ValueError(f"Unknown selection mode: {cfg.mode}")

        tau = float(getattr(cfg, "redundancy_tau", 0.0) or 0.0)
        base_M = M0.loc[:, order] if order else M0.iloc[:, :0]
        order = _redundancy_filter(base_M, order, tau)
        cols = [c for c in order if c in M.columns]

    # optional whitelist intersection from SIS-ΔRMSE
    wl_path = getattr(cfg, "sis_whitelist_path", None)
    if wl_path:
        try:
            if wl_path.endswith(".json"):
                wl = pd.read_json(wl_path, typ="series").tolist()
            else:
                wl = pd.read_csv(wl_path, header=None).iloc[:,0].astype(str).tolist()
            cols = [c for c in cols if c in wl]
        except Exception:
            pass

    if len(cols) == 0:
        raise ValueError("No engineered columns selected.")
    return cols
