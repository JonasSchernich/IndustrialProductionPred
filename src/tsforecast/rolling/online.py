from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import sys
import numpy as np
import pandas as pd

from ..types import FeatureSelectCfg, FeEngCfg
from ..features.selection import select_engineered_features
from ..features.engineering import (
    make_per_feature_lags_by_corr, build_engineered_matrix, apply_pca_train_transform
)
from ..models.registry import build_estimator
from ..tuning.grid import expand_grid

# Switch PCA location globally from the notebook: "post" (default) or "pre"
PCA_STAGE_DEFAULT = "post"

BASELINE_MODELS = {"mean", "avg", "average", "randomwalk", "rw", "naive", "ar1", "ar(1)"}

def _align_after_engineering(M: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop leading rows until BOTH M-row has no NaNs AND y is not NaN,
    then drop any residual invalid rows. Allows keeping January features
    even if the first y-change is NaN.
    """
    mask_valid = (~M.isna().any(axis=1)) & y.notna()
    if not mask_valid.any():
        return M.iloc[0:0, :], y.iloc[0:0]
    first = int(np.argmax(mask_valid.values))
    M2, y2 = M.iloc[first:, :], y.iloc[first:]
    tail = (~M2.isna().any(axis=1)) & y2.notna()
    return M2.loc[tail], y2.loc[tail]

def score_config_for_next_step(
    X: pd.DataFrame, y: pd.Series, t: int,
    base_features: List[str], fe_spec: Dict,
    fs_cfg: FeatureSelectCfg,
    model_name: str, model_params: Dict, metric_fn
) -> Tuple[float, float, Optional[List[str]]]:
    # Baselines ignore X
    if model_name.lower() in BASELINE_MODELS:
        ytr = y.iloc[:t+1]
        est = build_estimator(model_name, dict(model_params))
        est.fit(None, ytr)
        yhat = float(np.asarray(est.predict([[0]]))[0])
        val = float(metric_fn([y.iloc[t+1]], [yhat]))
        return val, yhat, None

    # PCA location
    pca_n   = fe_spec.get("pca_n")   if isinstance(fe_spec, dict) else None
    pca_var = fe_spec.get("pca_var") if isinstance(fe_spec, dict) else None
    pca_stage = (fe_spec.get("pca_stage") or PCA_STAGE_DEFAULT).lower() if isinstance(fe_spec, dict) else PCA_STAGE_DEFAULT

    # slices
    Xtr = X.iloc[:t+1, :][base_features]
    Xev = X.iloc[:t+2, :][base_features]  # last row = t+1

    if pca_stage == "pre" and (pca_n is not None or pca_var is not None):
        # PRE-PCA: basis -> PCs -> engineer (lags/rm/ema) on PCs
        PCtr, PCev = apply_pca_train_transform(Xtr, Xev, pca_n=pca_n, pca_var=pca_var)
        pc_cols = list(PCtr.columns)
        Mtr = build_engineered_matrix(PCtr, pc_cols, fe_spec)
        Mev_full = build_engineered_matrix(PCev, pc_cols, fe_spec).iloc[[-1], :]
        if Mtr.shape[1] == 0:
            raise ValueError("Engineered matrix has 0 columns.")
        Mtr2, ytr2 = _align_after_engineering(Mtr, y.iloc[:t+1])
        eng_cols = select_engineered_features(Mtr2, ytr2, fs_cfg)
        if len(eng_cols) == 0:
            raise ValueError("No engineered columns selected.")
        Mtr_fin = Mtr2[eng_cols]
        Mev_fin = Mev_full[eng_cols]
    else:
        # POST-PCA: basis -> engineer -> selection -> optional PCA
        Mtr = build_engineered_matrix(Xtr, base_features, fe_spec)
        Mev_full = build_engineered_matrix(Xev, base_features, fe_spec).iloc[[-1], :]
        if Mtr.shape[1] == 0:
            raise ValueError("Engineered matrix has 0 columns.")
        Mtr2, ytr2 = _align_after_engineering(Mtr, y.iloc[:t+1])
        eng_cols = select_engineered_features(Mtr2, ytr2, fs_cfg)
        if len(eng_cols) == 0:
            raise ValueError("No engineered columns selected.")
        Mtr_sel = Mtr2[eng_cols]
        Mev_sel = Mev_full[eng_cols]
        Mtr_fin, Mev_fin = apply_pca_train_transform(Mtr_sel, Mev_sel, pca_n=pca_n, pca_var=pca_var)

    est = build_estimator(model_name, dict(model_params))
    est.fit(Mtr_fin, ytr2)
    yhat = float(np.asarray(est.predict(Mev_fin))[0])
    val = float(metric_fn([y.iloc[t+1]], [yhat]))
    return val, yhat, eng_cols

def _fe_candidates_from_cfg(Xtr, ytr, base_features, fe_cfg: FeEngCfg):
    """Build FE candidate specs. Propagate optional external paths (tsfresh/fm)."""
    specs = []
    extra = {k: getattr(fe_cfg, k) for k in ("tsfresh_path", "fm_pred_path") if hasattr(fe_cfg, k)}

    if fe_cfg.per_feature_lags:
        lag_map = make_per_feature_lags_by_corr(
            Xtr[base_features], ytr, fe_cfg.per_feature_candidates, fe_cfg.per_feature_topk
        )
        for rms in fe_cfg.candidate_rm_sets:
            for emas in fe_cfg.candidate_ema_sets:
                for pca_n, pca_var in fe_cfg.candidate_pca:
                    spec = {"lag_map": lag_map, "rm_windows": rms, "ema_spans": emas,
                            "pca_n": pca_n, "pca_var": pca_var, **extra}
                    specs.append(spec)
    else:
        for lset in fe_cfg.candidate_lag_sets:
            for rms in fe_cfg.candidate_rm_sets:
                for emas in fe_cfg.candidate_ema_sets:
                    for pca_n, pca_var in fe_cfg.candidate_pca:
                        spec = {"lags": lset, "rm_windows": rms, "ema_spans": emas,
                                "pca_n": pca_n, "pca_var": pca_var, **extra}
                        specs.append(spec)
    return specs

def _count_evals(n_hp: int, n_fe: int, full: bool) -> int:
    return n_hp * n_fe if full else n_hp + (n_fe - 1)

def online_rolling_forecast(
    X: pd.DataFrame, y: pd.Series,
    initial_window: int, step: int, horizon: int,
    fs_cfg: FeatureSelectCfg, fe_cfg: FeEngCfg,
    model_name: str, model_grid: Dict, metric_fn: Callable,
    progress: bool = False, progress_fn: Optional[Callable] = None
):
    """Walk-forward: best-init, then one-step delayed search each step."""

    base_features0 = list(X.columns)
    start_t = initial_window - 1

    Xtr0, ytr0 = X.iloc[:start_t+1, :], y.iloc[:start_t+1]
    fe_candidates = _fe_candidates_from_cfg(Xtr0, ytr0, base_features0, fe_cfg)

    hp_list = list(expand_grid(model_grid))
    n_hp = len(hp_list)
    n_fe_init = len(fe_candidates)
    init_evals = _count_evals(n_hp, n_fe_init, fe_cfg.optimize_fe_for_all_hp)

    def _notify(stage: str, info: Dict):
        if progress_fn is not None:
            try:
                progress_fn(stage, info)
            except Exception:
                pass
        if progress:
            print(f"[{stage}] " + ", ".join(f"{k}={v}" for k, v in info.items())); sys.stdout.flush()

    _notify("init_start", {"n_hp": n_hp, "n_fe": n_fe_init, "expected_evals": init_evals})

    best_init = None
    evals_done = 0
    for i_hp, mp in enumerate(hp_list, start=1):
        for i_fe, fe_spec in enumerate(fe_candidates, start=1):
            score, yhat, _ = score_config_for_next_step(
                X, y, start_t, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn
            )
            evals_done += 1
            _notify("init_eval", {"done": evals_done, "total": init_evals, "hp_idx": i_hp, "fe_idx": i_fe, "score": round(score, 6)})
            cand = {"t": start_t, "fe_spec": fe_spec, "model_params": mp, "score": score, "yhat": yhat}
            if (best_init is None) or (score < best_init["score"]):
                best_init = cand

    _notify("init_done", {"best_score": round(best_init["score"], 6)})

    preds, truths, config_log = [], [], []
    prev_best = best_init

    for t0 in range(start_t, len(y)-1, step):
        t_pred = t0 + horizon

        # predict with previous best
        used_cols = None
        try:
            score_used, yhat, used_cols = score_config_for_next_step(
                X, y, t0, base_features0, prev_best["fe_spec"], fs_cfg, model_name, prev_best["model_params"], metric_fn
            )
            preds.append((y.index[t_pred], yhat))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))
            _notify("step_predict", {"t": int(t0), "yhat": round(yhat, 6)})
        except Exception as e:
            _notify("step_predict", {"t": int(t0), "error": str(e)})
            preds.append((y.index[t_pred], np.nan))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))

        # search next best
        _notify("step_search_start", {"t": int(t0)})
        best_now = None
        for mp in hp_list:
            fe_list = fe_candidates if fe_cfg.optimize_fe_for_all_hp else [prev_best["fe_spec"]]
            for fe_spec in fe_list:
                sc, yh, _ = score_config_for_next_step(
                    X, y, t0, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn
                )
                cand = {"t": t0, "fe_spec": fe_spec, "model_params": mp, "score": sc, "yhat": yh}
                if (best_now is None) or (sc < best_now["score"]):
                    best_now = cand
        _notify("step_done", {"best_score": round(best_now["score"], 6)})
        config_log.append({
            "time_for_pred": y.index[t_pred],
            "used_model_params": prev_best["model_params"],
            "used_fe_spec": prev_best["fe_spec"],
            "n_engineered_cols_used": None if used_cols is None else len(used_cols),
            "selected_next_score": best_now["score"],
        })
        prev_best = best_now

    preds = pd.Series({ts: val for ts, val in preds}).sort_index()
    truths = pd.Series({ts: val for ts, val in truths}).sort_index()
    cfgdf = pd.DataFrame(config_log).set_index("time_for_pred").sort_index()
    _notify("done", {"n_preds": len(preds)})
    return preds, truths, cfgdf
