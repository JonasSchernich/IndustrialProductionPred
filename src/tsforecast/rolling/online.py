# rolling/online.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import sys
from ..tuning.grid import expand_grid
from ..types import FeatureSelectCfg
from ..features.selection import select_engineered_features
from ..features.engineering import (
    build_engineered_matrix, apply_pca_train_transform
)
from ..models.registry import build_estimator

BASELINE_MODELS = {"mean", "avg", "average", "randomwalk", "rw", "naive", "ar1", "ar(1)"}
PCA_STAGE_DEFAULT = "post"  # "pre" oder "post" – vom Notebook aus umschaltbar

def _align_after_engineering(M: pd.DataFrame, y: pd.Series):
    mask = ~M.isna().any(axis=1)
    if not mask.any():
        return M.iloc[0:0], y.iloc[0:0]
    first = int(np.argmax(mask.values))
    return M.iloc[first:, :], y.iloc[first:]

def score_config_for_next_step(
    X: pd.DataFrame, y: pd.Series, t: int,
    base_features: List[str], fe_spec: Dict,
    fs_cfg: FeatureSelectCfg,
    model_name: str, model_params: Dict, metric_fn
) -> Tuple[float, float, Optional[List[str]]]:
    # Baselines
    if model_name.lower() in BASELINE_MODELS:
        ytr = y.iloc[:t+1]
        est = build_estimator(model_name, dict(model_params))
        est.fit(None, ytr)
        yhat = float(np.asarray(est.predict([[0]]))[0])
        val = float(metric_fn([y.iloc[t+1]], [yhat]))
        return val, yhat, None

    # PCA-Config
    pca_n   = fe_spec.get("pca_n")   if isinstance(fe_spec, dict) else None
    pca_var = fe_spec.get("pca_var") if isinstance(fe_spec, dict) else None
    # Wahl: fe_spec['pca_stage'] hat Vorrang, sonst globaler Default
    try:
        pca_stage = (fe_spec.get("pca_stage") or PCA_STAGE_DEFAULT).lower()
    except Exception:
        pca_stage = str(PCA_STAGE_DEFAULT).lower()

    # Daten-Slices
    Xtr = X.iloc[:t+1, :][base_features]
    Xev = X.iloc[:t+2, :][base_features]  # letzte Zeile = t+1

    if pca_stage == "pre" and (pca_n is not None or pca_var is not None):
        # PRE-PCA: Basis -> PCs -> (darauf) Lags/RM/EMA
        PCtr, PCev = apply_pca_train_transform(Xtr, Xev, pca_n=pca_n, pca_var=pca_var)
        pc_cols = list(PCtr.columns)

        Mtr = build_engineered_matrix(PCtr, pc_cols, fe_spec)
        Mev_full = build_engineered_matrix(PCev, pc_cols, fe_spec).iloc[[-1], :]
        if Mtr.shape[1] == 0:
            raise ValueError("Engineered matrix has 0 columns. Check FE spec.")

        Mtr2, ytr2 = _align_after_engineering(Mtr, y.iloc[:t+1])
        eng_cols = select_engineered_features(Mtr2, ytr2, fs_cfg)
        if len(eng_cols) == 0:
            raise ValueError("No engineered columns selected. Adjust fs_cfg.")

        Mtr_fin = Mtr2[eng_cols]
        Mev_fin = Mev_full[eng_cols]

    else:
        # POST-PCA (Standard): Basis -> Lags/RM/EMA -> Selektion -> (optional) PCA
        Mtr = build_engineered_matrix(Xtr, base_features, fe_spec)
        Mev_full = build_engineered_matrix(Xev, base_features, fe_spec).iloc[[-1], :]
        if Mtr.shape[1] == 0:
            raise ValueError("Engineered matrix has 0 columns. Check FE spec.")

        Mtr2, ytr2 = _align_after_engineering(Mtr, y.iloc[:t+1])
        eng_cols = select_engineered_features(Mtr2, ytr2, fs_cfg)
        if len(eng_cols) == 0:
            raise ValueError("No engineered columns selected. Adjust fs_cfg.")

        Mtr_sel = Mtr2[eng_cols]
        Mev_sel = Mev_full[eng_cols]
        Mtr_fin, Mev_fin = apply_pca_train_transform(Mtr_sel, Mev_sel, pca_n=pca_n, pca_var=pca_var)

    est = build_estimator(model_name, dict(model_params))
    est.fit(Mtr_fin, ytr2)
    yhat = float(np.asarray(est.predict(Mev_fin))[0])
    val = float(metric_fn([y.iloc[t+1]], [yhat]))
    return val, yhat, eng_cols



def _fe_candidates_from_cfg(Xtr, ytr, base_features, fe_cfg: FeEngCfg):
    specs = []
    stages = getattr(fe_cfg, "pca_stage_options", ("post",))
    for stage in stages:
        for pca_n, pca_var in fe_cfg.candidate_pca:
            if fe_cfg.per_feature_lags:
                if stage == "pre" and (pca_n is not None or pca_var is not None):
                    # PCA auf Basisfeatures fitten -> auf Train transformieren
                    Mtr_p, _ = apply_pca_train_transform(Xtr[base_features], Xtr[base_features],
                                                         pca_n=pca_n, pca_var=pca_var)
                    lag_map = make_per_feature_lags_by_corr(Mtr_p, ytr,
                                                            fe_cfg.per_feature_candidates,
                                                            fe_cfg.per_feature_topk)
                    for rms in fe_cfg.candidate_rm_sets:
                        for emas in fe_cfg.candidate_ema_sets:
                            specs.append({"pca_stage": "pre", "pca_n": pca_n, "pca_var": pca_var,
                                          "lag_map": lag_map, "rm_windows": rms, "ema_spans": emas})
                else:
                    lag_map = make_per_feature_lags_by_corr(
                        Xtr[base_features], ytr, fe_cfg.per_feature_candidates, fe_cfg.per_feature_topk
                    )
                    for rms in fe_cfg.candidate_rm_sets:
                        for emas in fe_cfg.candidate_ema_sets:
                            specs.append({"pca_stage": "post", "pca_n": pca_n, "pca_var": pca_var,
                                          "lag_map": lag_map, "rm_windows": rms, "ema_spans": emas})
            else:
                for lset in fe_cfg.candidate_lag_sets:
                    for rms in fe_cfg.candidate_rm_sets:
                        for emas in fe_cfg.candidate_ema_sets:
                            specs.append({"pca_stage": ("pre" if stage == "pre" else "post"),
                                          "pca_n": pca_n, "pca_var": pca_var,
                                          "lags": lset, "rm_windows": rms, "ema_spans": emas})
    return specs


def _count_evals(n_hp: int, n_fe: int, optimize_fe_for_all_hp: bool) -> int:
    if n_hp <= 0 or n_fe <= 0:
        return 0
    if optimize_fe_for_all_hp:
        return n_hp * n_fe
    # first HP tries all FE, others try one FE
    return n_fe + max(0, n_hp - 1)

def online_rolling_forecast(
    X: pd.DataFrame, y: pd.Series,
    initial_window: int, step: int, horizon: int,
    fs_cfg: FeatureSelectCfg, fe_cfg: FeEngCfg,
    model_name: str, model_grid: Dict, metric_fn,
    progress: bool = True,
    progress_fn: Optional[Callable[[str, Dict], None]] = None,
):
    assert horizon == 1, "This implementation currently supports horizon=1."
    idx = X.index
    n = len(idx)
    start_t = initial_window - 1
    max_t = n - 2
    total_steps = ((max_t - (start_t + step)) // step + 1) if max_t >= start_t + step else 0

    base_features0 = list(X.columns)

    Xtr0 = X.iloc[:initial_window, :]
    ytr0 = y.iloc[:initial_window]

    if model_name.lower() in BASELINE_MODELS:
        fe_candidates = [{"pca_n": None, "pca_var": None}]
    else:
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
            msg = f"[{stage}] " + ", ".join(f"{k}={v}" for k, v in info.items())
            print(msg)
            sys.stdout.flush()

    _notify("init_start", {"n_hp": n_hp, "n_fe": n_fe_init, "expected_evals": init_evals})

    best_init = None
    evals_done = 0
    for i_hp, mp in enumerate(hp_list, start=1):
        fe_list = fe_candidates if fe_cfg.optimize_fe_for_all_hp or best_init is None else [best_init["fe_spec"]]
        for i_fe, fe_spec in enumerate(fe_list, start=1):
            score, yhat, _ = score_config_for_next_step(
                X, y, start_t, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn
            )
            evals_done += 1
            _notify("init_eval", {"done": evals_done, "total": init_evals, "hp_idx": i_hp, "fe_idx": i_fe, "score": round(score, 6)})
            cand = {"t": start_t, "fe_spec": fe_spec, "model_params": mp, "score": score, "yhat": yhat}
            if (best_init is None) or (score < best_init["score"]):
                best_init = cand

    _notify("init_done", {"best_score": round(best_init['score'], 6)})

    prev_best = best_init
    preds, truths, config_log = [], [], []

    step_counter = 0
    for t in range(start_t + step, max_t + 1, step):
        step_counter += 1
        _notify("step_predict", {"step": step_counter, "of": total_steps, "date": str(idx[t+1].date())})

        score_tmp, yhat, used_cols = score_config_for_next_step(
            X, y, t, base_features0, prev_best["fe_spec"], fs_cfg, model_name, prev_best["model_params"], metric_fn
        )
        preds.append((idx[t+1], yhat))
        truths.append((idx[t+1], float(y.iloc[t+1])))

        if model_name.lower() in BASELINE_MODELS:
            fe_candidates_now = [{"pca_n": None, "pca_var": None}]
        else:
            Xtr = X.iloc[:t+1, :]
            ytr = y.iloc[:t+1]
            fe_candidates_now = _fe_candidates_from_cfg(Xtr, ytr, base_features0, fe_cfg)

        n_fe_now = len(fe_candidates_now)
        step_evals_total = _count_evals(n_hp, n_fe_now, fe_cfg.optimize_fe_for_all_hp)
        _notify("step_search_start", {"step": step_counter, "hp": n_hp, "fe": n_fe_now, "expected_evals": step_evals_total})

        best_now = None
        evals_done = 0
        for i_hp, mp in enumerate(hp_list, start=1):
            fe_list = fe_candidates_now if fe_cfg.optimize_fe_for_all_hp or best_now is None else [best_now["fe_spec"]]
            for i_fe, fe_spec in enumerate(fe_list, start=1):
                score, yhat_sel, _ = score_config_for_next_step(
                    X, y, t, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn
                )
                evals_done += 1
                if step_evals_total > 0:
                    _notify("step_eval", {"step": step_counter, "done": evals_done, "total": step_evals_total,
                                          "hp_idx": i_hp, "fe_idx": i_fe, "score": round(score, 6)})
                cand = {"t": t, "fe_spec": fe_spec, "model_params": mp, "score": score, "yhat": yhat_sel}
                if (best_now is None) or (score < best_now["score"]):
                    best_now = cand

        config_log.append({
            "time_for_pred": idx[t+1],
            "used_model_params": prev_best["model_params"],
            "used_fe_spec": prev_best["fe_spec"],
            "n_engineered_cols_used": None if used_cols is None else len(used_cols),
            "selected_next_score": best_now["score"],
        })
        prev_best = best_now

        _notify("step_done", {"step": step_counter, "best_score": round(best_now['score'], 6)})

    preds = pd.Series({ts: val for ts, val in preds}).sort_index()
    truths = pd.Series({ts: val for ts, val in truths}).sort_index()
    cfgdf = pd.DataFrame(config_log).set_index("time_for_pred").sort_index()

    _notify("done", {"n_preds": len(preds)})
    return preds, truths, cfgdf
