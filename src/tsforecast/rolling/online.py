from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import sys
import numpy as np
import pandas as pd

from ..types import FeatureSelectCfg, FeEngCfg, TrainEvalCfg, AshaCfg
from ..features.selection import select_engineered_features
from ..features.engineering import (
    make_per_feature_lags_by_corr, build_engineered_matrix, apply_pca_train_transform
)
from ..models.registry import build_estimator, supports_es
from ..tuning.grid import expand_grid

PCA_STAGE_DEFAULT = "post"
BASELINE_MODELS = {"mean", "avg", "average", "randomwalk", "rw", "naive", "ar1", "ar(1)"}


def _align_after_engineering(M: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
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
    model_name: str, model_params: Dict, metric_fn,
    train_eval_cfg: Optional[TrainEvalCfg] = None
) -> Tuple[float, float, Optional[List[str]]]:
    if model_name.lower() in BASELINE_MODELS:
        ytr = y.iloc[:t+1]
        est = build_estimator(model_name, dict(model_params))
        est.fit(None, ytr)
        yhat = float(np.asarray(est.predict([[0]]))[0])
        val = float(metric_fn([y.iloc[t+1]], [yhat]))
        return val, yhat, None

    pca_n   = fe_spec.get("pca_n")   if isinstance(fe_spec, dict) else None
    pca_var = fe_spec.get("pca_var") if isinstance(fe_spec, dict) else None
    pca_stage = (fe_spec.get("pca_stage") or PCA_STAGE_DEFAULT).lower() if isinstance(fe_spec, dict) else PCA_STAGE_DEFAULT

    Xtr = X.iloc[:t+1, :][base_features]
    Xev = X.iloc[:t+2, :][base_features]

    if pca_stage == "pre" and (pca_n is not None or pca_var is not None):
        PCtr, PCev = apply_pca_train_transform(Xtr, Xev, pca_n=pca_n, pca_var=pca_var, pca_solver=fe_spec.get("pca_solver","auto"))
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
        Mtr_fin, Mev_fin = apply_pca_train_transform(Mtr_sel, Mev_sel, pca_n=pca_n, pca_var=pca_var, pca_solver=fe_spec.get("pca_solver","auto"))

    est = build_estimator(model_name, dict(model_params))

    # ES nur falls konfiguriert und unterstützt
    use_es = False
    if train_eval_cfg is not None and train_eval_cfg.early_stopping_rounds > 0 and supports_es(model_name):
        dev_tail = max(0, int(train_eval_cfg.dev_tail))
        if dev_tail > 0 and len(Mtr_fin) > dev_tail:
            X_train = Mtr_fin.iloc[:-dev_tail, :]
            y_train = ytr2.iloc[:-dev_tail]
            X_dev = Mtr_fin.iloc[-dev_tail:, :]
            y_dev = ytr2.iloc[-dev_tail:]
            try:
                est.fit(
                    X_train, y_train,
                    eval_set=[(X_dev, y_dev)],
                    early_stopping_rounds=int(train_eval_cfg.early_stopping_rounds)
                )
                use_es = True
            except Exception:
                est.fit(Mtr_fin, ytr2)
        else:
            est.fit(Mtr_fin, ytr2)
    else:
        est.fit(Mtr_fin, ytr2)

    yhat = float(np.asarray(est.predict(Mev_fin))[0])
    val = float(metric_fn([y.iloc[t+1]], [yhat]))
    return val, yhat, eng_cols


def _fe_candidates_from_cfg(Xtr, ytr, base_features, fe_cfg, existing_lag_map: Optional[dict] = None):
    specs, extra = [], {k: getattr(fe_cfg, k) for k in ("tsfresh_path","fm_pred_path") if hasattr(fe_cfg,k)}
    if fe_cfg.per_feature_lags:
        lag_map = existing_lag_map or make_per_feature_lags_by_corr(
            Xtr[base_features], ytr, fe_cfg.per_feature_candidates, fe_cfg.per_feature_topk
        )
        for rms in fe_cfg.candidate_rm_sets:
            for emas in fe_cfg.candidate_ema_sets:
                for pca_n, pca_var in fe_cfg.candidate_pca:
                    for pca_stage in fe_cfg.pca_stage_options:
                        specs.append({
                            "lag_map": lag_map,
                            "rm_windows": rms,
                            "ema_spans": emas,
                            "pca_n": pca_n,
                            "pca_var": pca_var,
                            "pca_stage": pca_stage,
                            "pca_solver": getattr(fe_cfg, "pca_solver", "auto"),
                            **extra
                        })
    else:
        for lset in fe_cfg.candidate_lag_sets:
            for rms in fe_cfg.candidate_rm_sets:
                for emas in fe_cfg.candidate_ema_sets:
                    for pca_n, pca_var in fe_cfg.candidate_pca:
                        for pca_stage in fe_cfg.pca_stage_options:
                            specs.append({
                                "lags": lset,
                                "rm_windows": rms,
                                "ema_spans": emas,
                                "pca_n": pca_n,
                                "pca_var": pca_var,
                                "pca_stage": pca_stage,
                                "pca_solver": getattr(fe_cfg, "pca_solver", "auto"),
                                **extra
                            })
    return specs


def _count_evals(n_hp: int, n_fe: int, full: bool) -> int:
    return n_hp * n_fe if full else n_hp + (n_fe - 1)


def _notify(progress: bool, progress_fn: Optional[Callable], stage: str, info: Dict):
    if progress_fn is not None:
        try:
            progress_fn(stage, info)
        except Exception:
            pass
    if progress:
        print(f"[{stage}] " + ", ".join(f"{k}={v}" for k, v in info.items())); sys.stdout.flush()


def _sample_candidates(hp_list: List[Dict], fe_list: List[Dict], n: int, rng: np.random.RandomState):
    pairs = [(hp, fe) for hp in hp_list for fe in fe_list]
    if n >= len(pairs):
        return pairs
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[i] for i in idx]


def _eval_over_steps(
    X, y, base_features, fs_cfg, model_name, hp, fe_spec,
    metric_fn, train_eval_cfg: Optional[TrainEvalCfg],
    t_start: int, steps: int, step_incr: int
) -> float:
    scores = []
    n = len(y)
    end_t = min(t_start + steps - 1, n - 2)
    for t in range(t_start, end_t + 1, step_incr):
        try:
            sc, _, _ = score_config_for_next_step(
                X, y, t, base_features, fe_spec, fs_cfg, model_name, hp, metric_fn, train_eval_cfg
            )
            scores.append(sc)
        except Exception:
            continue
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _asha_initial_search(
    X: pd.DataFrame, y: pd.Series, base_features: List[str],
    fs_cfg: FeatureSelectCfg, fe_cfg: FeEngCfg,
    model_name: str, hp_list: List[Dict], fe_candidates: List[Dict],
    metric_fn: Callable, train_eval_cfg: Optional[TrainEvalCfg],
    asha_cfg: AshaCfg, start_t: int, step_incr: int,
    progress: bool, progress_fn: Optional[Callable]
):
    rng = np.random.RandomState(None if asha_cfg.seed is None else int(asha_cfg.seed))

    def stage_eval(pairs, steps, tag):
        best = []
        for i, (hp, fe) in enumerate(pairs, start=1):
            sc = _eval_over_steps(X, y, base_features, fs_cfg, model_name, hp, fe, metric_fn, train_eval_cfg, start_t, steps, step_incr)
            _notify(progress, progress_fn, "asha_eval", {"stage": tag, "i": i, "score": round(sc, 6)})
            best.append((sc, hp, fe))
        best.sort(key=lambda x: x[0])
        return best

    # B1
    b1_pairs = _sample_candidates(hp_list, fe_candidates, int(asha_cfg.n_b1), rng)
    res1 = stage_eval(b1_pairs, int(asha_cfg.steps_b1), "B1")
    k1 = max(1, int(len(res1) * float(asha_cfg.promote_frac_1)))
    top1 = res1[:k1]

    # B2
    b2_pairs = [(hp, fe) for _, hp, fe in top1]
    b2_pairs = b2_pairs[: int(asha_cfg.n_b2)] if len(b2_pairs) > int(asha_cfg.n_b2) else b2_pairs
    res2 = stage_eval(b2_pairs, int(asha_cfg.steps_b2), "B2")
    k2 = max(1, int(len(res2) * float(asha_cfg.promote_frac_2)))
    top2 = res2[:k2]

    # B3
    b3_pairs = [(hp, fe) for _, hp, fe in top2]
    b3_pairs = b3_pairs[: int(asha_cfg.n_b3)] if len(b3_pairs) > int(asha_cfg.n_b3) else b3_pairs
    res3 = stage_eval(b3_pairs, int(asha_cfg.steps_b3), "B3")

    best_sc, best_hp, best_fe = res3[0]
    return {"t": start_t, "fe_spec": best_fe, "model_params": best_hp, "score": best_sc, "yhat": None}


def online_rolling_forecast(
    X: pd.DataFrame, y: pd.Series,
    initial_window: int, step: int, horizon: int,
    fs_cfg: FeatureSelectCfg, fe_cfg: FeEngCfg,
    model_name: str, model_grid: Dict, metric_fn: Callable,
    progress: bool = False, progress_fn: Optional[Callable] = None,
    per_feature_lag_refresh_k: Optional[int] = None,
    train_eval_cfg: Optional[TrainEvalCfg] = None,
    asha_cfg: Optional[AshaCfg] = None
):
    if horizon != 1:
        raise NotImplementedError("Aktuell nur horizon=1.")

    base_features0 = list(X.columns)
    start_t = initial_window - 1

    Xtr0, ytr0 = X.iloc[:start_t+1, :], y.iloc[:start_t+1]
    fe_candidates = _fe_candidates_from_cfg(Xtr0, ytr0, base_features0, fe_cfg)

    hp_list = list(expand_grid(model_grid))
    n_hp = len(hp_list)
    n_fe_init = len(fe_candidates)
    init_evals = _count_evals(n_hp, n_fe_init, fe_cfg.optimize_fe_for_all_hp)

    _notify(progress, progress_fn, "init_start", {"n_hp": n_hp, "n_fe": n_fe_init, "expected_evals": init_evals})

    # Initialbesten via ASHA oder klassisch
    best_init = None
    if asha_cfg is not None and asha_cfg.use_asha:
        best_init = _asha_initial_search(
            X, y, base_features0, fs_cfg, fe_cfg, model_name,
            hp_list, fe_candidates, metric_fn, train_eval_cfg,
            asha_cfg, start_t, step, progress, progress_fn
        )
        _notify(progress, progress_fn, "init_done_asha", {"best_score": round(best_init["score"], 6)})
    else:
        evals_done = 0
        for i_hp, mp in enumerate(hp_list, start=1):
            for i_fe, fe_spec in enumerate(fe_candidates, start=1):
                try:
                    score, yhat, _ = score_config_for_next_step(
                        X, y, start_t, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn, train_eval_cfg
                    )
                except Exception as e:
                    _notify(progress, progress_fn, "init_eval_error", {"hp_idx": i_hp, "fe_idx": i_fe, "err": str(e)})
                    continue
                evals_done += 1
                _notify(progress, progress_fn, "init_eval", {"done": evals_done, "total": init_evals, "hp_idx": i_hp, "fe_idx": i_fe, "score": round(score, 6)})
                cand = {"t": start_t, "fe_spec": fe_spec, "model_params": mp, "score": score, "yhat": yhat}
                if (best_init is None) or (score < best_init["score"]):
                    best_init = cand
        _notify(progress, progress_fn, "init_done", {"best_score": round(best_init["score"], 6)})

    preds, truths, config_log = [], [], []
    prev_best = best_init
    cached_lag_map: Optional[dict] = None

    for i_step, t0 in enumerate(range(start_t, len(y)-1, step)):
        t_pred = t0 + horizon

        # predict with previous best
        used_cols = None
        try:
            score_used, yhat, used_cols = score_config_for_next_step(
                X, y, t0, base_features0, prev_best["fe_spec"], fs_cfg, model_name, prev_best["model_params"], metric_fn, train_eval_cfg
            )
            preds.append((y.index[t_pred], yhat))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))
            _notify(progress, progress_fn, "step_predict", {"t": int(t0), "yhat": round(yhat, 6)})
        except Exception as e:
            _notify(progress, progress_fn, "step_predict_error", {"t": int(t0), "error": str(e)})
            preds.append((y.index[t_pred], np.nan))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))

        # FE candidates refresh (Lag-Map zyklisch)
        use_existing = (
            fe_cfg.per_feature_lags
            and cached_lag_map is not None
            and per_feature_lag_refresh_k is not None
            and per_feature_lag_refresh_k > 0
            and (i_step % per_feature_lag_refresh_k != 0)
        )
        fe_candidates = _fe_candidates_from_cfg(
            X.iloc[:t0+1, :], y.iloc[:t0+1], base_features0, fe_cfg,
            existing_lag_map=(cached_lag_map if use_existing else None)
        )
        if fe_cfg.per_feature_lags and fe_candidates:
            lm = fe_candidates[0].get("lag_map")
            if lm is not None:
                cached_lag_map = lm

        # search next best (klassisch, schnell)
        _notify(progress, progress_fn, "step_search_start", {"t": int(t0)})
        best_now = None
        for mp in hp_list:
            fe_list = fe_candidates if fe_cfg.optimize_fe_for_all_hp else [prev_best["fe_spec"]]
            for fe_spec in fe_list:
                try:
                    sc, yh, _ = score_config_for_next_step(
                        X, y, t0, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn, train_eval_cfg
                    )
                except Exception as e:
                    _notify(progress, progress_fn, "candidate_error", {"t": int(t0), "err": str(e)})
                    continue
                cand = {"t": t0, "fe_spec": fe_spec, "model_params": mp, "score": sc, "yhat": yh}
                if (best_now is None) or (sc < best_now["score"]):
                    best_now = cand
        if best_now is None:
            raise RuntimeError(f"No valid candidate at step t={t0}")
        _notify(progress, progress_fn, "step_done", {"best_score": round(best_now["score"], 6)})
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
    _notify(progress, progress_fn, "done", {"n_preds": len(preds)})
    return preds, truths, cfgdf
