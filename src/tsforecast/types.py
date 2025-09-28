from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Feature Selection
@dataclass
class FeatureSelectCfg:
    """
    mode: "manual" | "auto_topk" | "auto_threshold" | "none"
    manual_cols: engineered column names to keep
    topk: used when mode='auto_topk'
    min_abs_corr: used when mode='auto_threshold'
    variance_thresh: drop columns with variance <= threshold (0.0 drops constant cols)
    """
    mode: str = "auto_topk"
    manual_cols: Optional[List[str]] = None
    topk: int = 200
    min_abs_corr: float = 0.2
    variance_thresh: float = 0.0

# Feature Engineering
@dataclass
class FeEngCfg:
    """
    FE search space and external blocks.
    - per_feature_lags: if True, use per-feature best lags via correlation
    - per_feature_candidates: e.g. (1,2,3,6,12)
    - per_feature_topk: per base feature keep top-k lags
    - candidate_lag_sets: tuple of lag-set candidates (when not per-feature)
    - candidate_rm_sets / candidate_ema_sets: tuples of rolling/ema windows
    - candidate_pca: tuple of (pca_n:int|None, pca_var:float|None)
    - pca_stage_options: ("pre","post")
    - optimize_fe_for_all_hp: Fast-Search toggle
    - tsfresh_path / fm_pred_path: optional parquet paths (shift(1) merge downstream)
    - pca_solver: "auto" | "full" | "randomized" (only effective for integer pca_n)
    """
    per_feature_lags: bool = False
    per_feature_candidates: Tuple[int, ...] = (1, 3, 6, 12)
    per_feature_topk: int = 1

    candidate_lag_sets: Tuple[Tuple[int, ...], ...] = ((1, 3, 6, 12),)
    candidate_rm_sets: Tuple[Tuple[int, ...], ...] = ((), (3,), (3, 6))
    candidate_ema_sets: Tuple[Tuple[int, ...], ...] = ((), (3,), (3, 6))
    candidate_pca: Tuple[Tuple[Optional[int], Optional[float]], ...] = ((None, None),)
    pca_stage_options: Tuple[str, ...] = ("post",)

    optimize_fe_for_all_hp: bool = True

    tsfresh_path: Optional[str] = None
    fm_pred_path: Optional[str] = None

    pca_solver: str = "auto"

# Training/Eval (ES)
@dataclass
class TrainEvalCfg:
    """
    dev_tail: last m months of train window used as dev for ES (m=0 disables split)
    early_stopping_rounds: 0 disables ES
    """
    dev_tail: int = 12
    early_stopping_rounds: int = 0

# ASHA config (light)
@dataclass
class AshaCfg:
    """
    use_asha: enable ASHA for initial search
    steps_b1/b2/b3: number of rolling steps to evaluate per stage
    n_b1/n_b2/n_b3: number of candidates evaluated per stage
    promote_frac_1/2: fraction to promote from B1->B2 and B2->B3 (0..1)
    seed: RNG seed for sampling
    """
    use_asha: bool = False
    steps_b1: int = 60
    steps_b2: int = 150
    steps_b3: int = 320
    n_b1: int = 60
    n_b2: int = 24
    n_b3: int = 8
    promote_frac_1: float = 0.4
    promote_frac_2: float = 0.3333
    seed: Optional[int] = 42

# Ensemble (placeholder for later stacking)
@dataclass
class EnsembleCfg:
    use_weighted_mean: bool = False
    use_stacking: bool = False
