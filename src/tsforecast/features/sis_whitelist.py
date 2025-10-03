import numpy as np, pandas as pd

def _rmse_np(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b)**2)))

def _ar1_fit(y):
    y = np.asarray(y, float)
    if len(y) < 2: return 0.0
    ylag, yt = y[:-1], y[1:]
    denom = float(np.dot(ylag, ylag))
    return float(np.dot(ylag, yt) / denom) if denom > 0 else 0.0

def sis_whitelist_from_calib(M: pd.DataFrame, y: pd.Series, features: list,
                             calib_start: int, calib_end: int,
                             baseline: str = "ar1", gain_thresh: float = 0.0) -> list:
    keep, T = [], len(y)
    lo, hi = max(1, int(calib_start)), min(T, int(calib_end))
    oos_idx = list(range(lo, hi))
    if not oos_idx: return features

    e_base = []
    for s in oos_idx:
        ytr = y.iloc[:s]
        if baseline == "ar1":
            phi = _ar1_fit(ytr.values)
            yhat = phi * float(ytr.iloc[-1])
        else:
            yhat = 0.0
        e_base.append(float(y.iloc[s] - yhat))
    rmse_base = _rmse_np(e_base, np.zeros_like(e_base))

    for c in features:
        ec = []
        x = M[c].values
        for s in oos_idx:
            xs = x[:s]; ys = y.iloc[:s].values
            xs = np.c_[np.ones((len(xs),1)), xs.reshape(-1,1)]
            try:
                b, *_ = np.linalg.lstsq(xs, ys, rcond=None)
                yhat = float(b[0] + b[1] * x[s-1])  # form at s-1 for s
            except:
                yhat = 0.0
            ec.append(float(y.iloc[s] - yhat))
        rmse_c = _rmse_np(ec, np.zeros_like(ec))
        if (rmse_base - rmse_c) >= float(gain_thresh):
            keep.append(c)
    return keep
