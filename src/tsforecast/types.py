
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class FeatureSelectCfg:
    """Configuration for selecting engineered features.
    - mode: "manual" | "auto_topk" | "auto_threshold"
    - manual_cols: engineered column names to keep (e.g., 'ifo_lag12', 'orders_rm3')
    - topk: number of engineered columns to keep when mode='auto_topk'
    - min_abs_corr: min |Pearson corr| threshold when mode='auto_threshold'
    - variance_thresh: drop engineered columns with variance <= this value
    """
    mode: str = "auto_topk"
    manual_cols: Optional[List[str]] = None
    topk: int = 200
    min_abs_corr: float = 0.0
    variance_thresh: float = 0.0

@dataclass
class LagCfg:
    """(Legacy) Lag-only config. Prefer FeEngCfg for full FE control."""
    candidate_lags: Tuple[Tuple[int, ...], ...] = ((1,3,6,12), (1,3,6), (1,12))
    per_feature: bool = False
    per_feature_candidates: Tuple[int, ...] = (1,3,6,12)
    per_feature_topk: int = 1
    optimize_lags_for_all_hp: bool = True

@dataclass
class FeEngCfg:
    """Full Feature Engineering configuration.
    - Candidate sets define alternative FE recipes that the search explores.
    - PCA: specify either pca_n (#components) or pca_var (explained variance) per candidate.
    """
    # Global lags for all base features
    candidate_lag_sets: Tuple[Tuple[int, ...], ...] = ((1,3,6,12), (1,3,6), (1,))
    # Rolling means (shifted by 1 to prevent leakage)
    candidate_rm_sets: Tuple[Tuple[int, ...], ...] = ((), (3,), (3,6))
    # EMAs (shifted by 1 to prevent leakage)
    candidate_ema_sets: Tuple[Tuple[int, ...], ...] = ((), (3,), (6,), (3,6))
    # PCA candidates: (pca_n, pca_var); exactly one should be non-None; (None, None) = no PCA
    candidate_pca: Tuple[Tuple[Optional[int], Optional[float]], ...] = ((None, None),)
    pca_stage_options: Tuple[str, ...] = ("post",)  # neu: "pre" oder "post"
    # Per-feature lagging: choose best k lags per base feature by |corr| to y in-train
    per_feature_lags: bool = False
    per_feature_candidates: Tuple[int, ...] = (1,3,6,12)
    per_feature_topk: int = 1
    # Search control
    optimize_fe_for_all_hp: bool = True
