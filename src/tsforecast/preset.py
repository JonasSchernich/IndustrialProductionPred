from __future__ import annotations
from typing import Tuple, Optional
from .types import FeEngCfg, FeatureSelectCfg, TrainEvalCfg, AshaCfg, BOCfg

# FE-Presets (familienweise)

def fe_cfg_tree(
    tsfresh: bool = False,
    fm_stack: bool = False,
    rm: Tuple[int, ...] = (3,),
    ema: Tuple[int, ...] = (3,),
    per_feature_candidates: Tuple[int, ...] = (1, 2, 3, 6, 12),
    per_feature_topk: int = 1,
) -> FeEngCfg:
    return FeEngCfg(
        per_feature_lags=True,
        per_feature_candidates=per_feature_candidates,
        per_feature_topk=per_feature_topk,
        candidate_lag_sets=((1, 3, 6, 12),),
        candidate_rm_sets=(tuple(), rm),
        candidate_ema_sets=(tuple(), ema),
        candidate_pca=((None, None),),          # Trees: keine PCA
        pca_stage_options=("post",),
        optimize_fe_for_all_hp=True,
        tsfresh_path="/mnt/data/tsfresh_slim.parquet" if tsfresh else None,
        fm_pred_path="/mnt/data/chronos_1step.parquet" if fm_stack else None,
        pca_solver="auto",
    )

def fe_cfg_linear(
    tsfresh: bool = False,
    fm_stack: bool = False,
    rm: Tuple[int, ...] = (3,),
    ema: Tuple[int, ...] = (3,),
) -> FeEngCfg:
    return FeEngCfg(
        per_feature_lags=False,
        candidate_lag_sets=((1, 3, 6, 12),),
        candidate_rm_sets=(tuple(), rm),
        candidate_ema_sets=(tuple(), ema),
        candidate_pca=((None, 0.95), (None, 0.99), (None, None)),  # PCA post
        pca_stage_options=("post",),
        optimize_fe_for_all_hp=True,
        tsfresh_path="/mnt/data/tsfresh_slim.parquet" if tsfresh else None,
        fm_pred_path="/mnt/data/chronos_1step.parquet" if fm_stack else None,
        pca_solver="auto",
    )

# Bequeme Defaults für FS/ES/ASHA/BO

def fs_cfg_default(
    mode: str = "auto_topk",
    topk: int = 250,
    variance_thresh: float = 0.0,
    min_abs_corr: float = 0.0,
    manual_cols: Optional[Tuple[str, ...]] = None,
) -> FeatureSelectCfg:
    return FeatureSelectCfg(
        mode=mode,
        topk=topk,
        variance_thresh=variance_thresh,
        min_abs_corr=min_abs_corr,
        manual_cols=list(manual_cols) if manual_cols else None,
    )

def train_eval_default(
    dev_tail: int = 12,
    early_stopping_rounds: int = 100,
) -> TrainEvalCfg:
    return TrainEvalCfg(dev_tail=dev_tail, early_stopping_rounds=early_stopping_rounds)

def asha_default(seed: int = 42) -> AshaCfg:
    return AshaCfg(
        use_asha=True, seed=seed,
        n_b1=60, steps_b1=60, promote_frac_1=0.25,
        n_b2=30, steps_b2=150, promote_frac_2=0.4,
        n_b3=8,  steps_b3=320
    )

def bo_default() -> BOCfg:
    return BOCfg(
        use_bo=True, n_iter=20, radius=0.25, steps=150, seed=123,
        hp_keys=("num_leaves", "min_child_samples", "reg_lambda", "colsample_bytree")
    )
