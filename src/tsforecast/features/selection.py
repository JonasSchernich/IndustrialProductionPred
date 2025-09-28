
from typing import List
import numpy as np
import pandas as pd
from ..types import FeatureSelectCfg

# selection.py
def _variance_filter(X: pd.DataFrame, thresh: float) -> pd.DataFrame:
    variances = X.var(axis=0)
    if thresh <= 0:
        keep = variances > 0.0
    else:
        keep = variances > float(thresh)
    return X.loc[:, keep]


def select_features(Xtr: pd.DataFrame, ytr: pd.Series, cfg: FeatureSelectCfg) -> List[str]:
    """(Legacy) Selection on BASE features (kept for compatibility in other scripts)."""
    Xtr2 = _variance_filter(Xtr, cfg.variance_thresh)
    if cfg.mode == "manual":
        cols = [c for c in (cfg.manual_cols or []) if c in Xtr2.columns]
        return cols
    corr = Xtr2.apply(lambda s: s.corr(ytr), axis=0).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if cfg.mode == "auto_topk":
        return corr.sort_values(ascending=False).head(cfg.topk).index.tolist()
    if cfg.mode == "auto_threshold":
        return corr[corr >= cfg.min_abs_corr].index.tolist()
    raise ValueError(f"Unknown selection mode: {cfg.mode}")

def select_engineered_features(Mtr, ytr, cfg):
    M2 = _variance_filter(Mtr, cfg.variance_thresh)
    if cfg.mode == "none":
        return list(M2.columns)
    if cfg.mode == "manual":
        return [c for c in (cfg.manual_cols or []) if c in M2.columns]
    corr = M2.corrwith(ytr).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if cfg.mode == "auto_topk":
        return corr.sort_values(ascending=False).head(cfg.topk).index.tolist()
    if cfg.mode == "auto_threshold":
        return corr[corr >= cfg.min_abs_corr].index.tolist()
    raise ValueError(f"Unknown selection mode: {cfg.mode}")

