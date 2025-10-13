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
def _residualize(M: pd.DataFrame, y: pd.Series, use_month_dummies: bool, use_y_lags: bool, y_lags=(1,)):
    y_col = y.name or "__y__"
    y_ = y.rename(y_col)

    Z = []
    if use_month_dummies:
        Z.append(_month_dummies(y_.index))
    if use_y_lags and y_lags:
        Z.append(pd.DataFrame({f"yl{L}": y_.shift(L).values for L in y_lags}, index=y_.index))

    Z = [z for z in Z if z is not None and z.shape[1] > 0]
    if len(Z) == 0:
        return M, y_

    Z = pd.concat(Z, axis=1).astype(float)
    A = pd.concat([M, y_, Z], axis=1).dropna()
    if A.empty:
        return M.iloc[0:0, :], y_.iloc[0:0]

    cols_M = list(M.columns)
    X = A[cols_M].astype(float).values
    z = np.c_[np.ones((len(A),1)), A[Z.columns].astype(float).values]
    yy = A[y_col].astype(float).values

    bz, *_ = np.linalg.lstsq(z, yy, rcond=None)
    ry = yy - z @ bz
    Bz, *_ = np.linalg.lstsq(z, X, rcond=None)
    RX = X - z @ Bz

    return pd.DataFrame(RX, index=A.index, columns=cols_M), pd.Series(ry, index=A.index, name=y_col)



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
    M = _variance_filter(Mtr, cfg.variance_thresh).astype(float)

    if cfg.mode == "none":
        cols = list(M.columns)
    elif cfg.mode == "manual":
        cols = [c for c in (cfg.manual_cols or []) if c in M.columns]
    else:
        if getattr(cfg, "prewhiten", False):
            lags = getattr(cfg, "y_lags", (1,))
            RX, Ry = _residualize(M, ytr, cfg.use_month_dummies, cfg.use_y_lags, lags)
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
def prewhitened_corr_screen(
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    topk: int = 400,
    min_abs_corr: float = 0.0,
    use_seasonal: bool = False,
) -> list:
    # D_s = (1, y_{s-1}, optional y_{s-12}) auf Trainingsfenster
    ylag1 = ytr.shift(1)
    cols = [np.ones(len(ytr))]
    if ylag1.notna().any():
        cols.append(ylag1.values)
    if use_seasonal:
        ylag12 = ytr.shift(12)
        cols.append(ylag12.values)
    D = np.column_stack(cols)
    ok_y = np.isfinite(ytr.shift(-1).values) & np.all(np.isfinite(D), axis=1)

    # y_{s+1} auf D_s residualisieren
    Y = ytr.shift(-1).values
    Dy = D[ok_y]; Yy = Y[ok_y]
    try:
        gy, *_ = np.linalg.lstsq(Dy, Yy, rcond=None)
        ry = np.empty_like(Y, dtype=float); ry[:] = np.nan
        ry[ok_y] = Yy - Dy.dot(gy)
    except:
        ry = (Y - np.nanmean(Y))

    scores = []
    for c in Xtr.columns:
        x = Xtr[c].values
        ok_x = np.isfinite(x) & np.all(np.isfinite(D), axis=1)
        ok = ok_x & np.isfinite(ry)
        if ok.sum() < 10:
            continue
        # x auf D residualisieren
        Dx = D[ok]; Xx = x[ok]
        try:
            gx, *_ = np.linalg.lstsq(Dx, Xx, rcond=None)
            rx = Xx - Dx.dot(gx)
        except:
            rx = Xx - Xx.mean()
        r = np.corrcoef(rx, ry[ok])[0, 1]
        if np.isfinite(r):
            scores.append((abs(float(r)), c))

    scores.sort(key=lambda z: z[0], reverse=True)
    if min_abs_corr > 0:
        scores = [s for s in scores if s[0] >= float(min_abs_corr)]
    keep = [c for _, c in (scores[:int(topk)] if topk else scores)]
    if not keep:
        raise ValueError("prewhitened_corr_screen: keine Features selektiert.")
    return keep
import numpy as np
import pandas as pd

def select_per_feature_lags_prewhite(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    lag_candidates=(1, 2, 3, 6, 12),
    topk=1,
    use_seasonal=False,
):
    # Residualisiere y_{s+1} auf D_s
    y_t1 = y.shift(-1)
    D = pd.DataFrame({"const": 1.0, "y1": y.shift(1)}, index=y.index)
    if use_seasonal:
        D["y12"] = y.shift(12)

    D_tr = D.loc[train_idx].dropna()
    y_tr = y_t1.loc[D_tr.index]

    # OLS-Projektion: r^y = y - D * (D'D)^{-1} D'y
    DtD = D_tr.to_numpy().T @ D_tr.to_numpy()
    DtD_inv = np.linalg.pinv(DtD)
    Py = D_tr.to_numpy() @ (DtD_inv @ (D_tr.to_numpy().T @ y_tr.to_numpy()))
    r_y = pd.Series(y_tr.to_numpy() - Py, index=D_tr.index)

    # Für jede Spalte und jeden Lag: Residualisiere x_{j,s} auf D_s und korreliere mit r_y (auf Schnittmenge)
    sel = {}
    for j in X.columns:
        scores = []
        for L in lag_candidates:
            xL = X[j].shift(L)
            # gleiche Zeilen wie D_tr
            x_tr = xL.loc[D_tr.index]
            # r^x = x - D * (D'D)^{-1} D'x
            Px = D_tr.to_numpy() @ (DtD_inv @ (D_tr.to_numpy().T @ x_tr.to_numpy()))
            r_x = pd.Series(x_tr.to_numpy() - Px, index=D_tr.index)
            # Korrelation auf gültigen Zeilen
            m = r_x.notna() & r_y.notna()
            if m.sum() < 8:
                continue
            corr = np.corrcoef(r_x[m].to_numpy(), r_y[m].to_numpy())[0, 1]
            scores.append((L, abs(corr)))
        if scores:
            scores.sort(key=lambda t: t[1], reverse=True)
            sel[j] = [L for (L, _) in scores[:topk]]
        else:
            sel[j] = []
    return sel
