from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import sys
import json
import hashlib
import numpy as np
import pandas as pd

from ..types import FeatureSelectCfg, FeEngCfg, TrainEvalCfg, AshaCfg, BOCfg
from ..features.selection import select_engineered_features
from ..features.engineering import (
    make_per_feature_lags_by_corr, build_engineered_matrix, apply_pca_train_transform
)
from ..models.registry import build_estimator, supports_es
from ..tuning.grid import expand_grid
from ..utils.reporting import append_summary_row, append_tuning_rows, append_predictions_rows

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

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(float)
    return float(np.sqrt(np.mean(d * d)))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs((a - b).astype(float))
    return float(np.mean(d))

def _notify(progress: bool, progress_fn: Optional[Callable], stage: str, info: Dict):
    if progress_fn is not None:
        try:
            progress_fn(stage, info)
        except Exception:
            pass
    if progress:
        print(f"[{stage}] " + ", ".join(f"{k}={v}" for k, v in info.items())); sys.stdout.flush()

def _summarize_fe_spec(fe_spec: Dict) -> Dict:
    return {
        "pca_stage": fe_spec.get("pca_stage"),
        "pca_n": fe_spec.get("pca_n"),
        "pca_var": fe_spec.get("pca_var"),
        "per_feature_lags": bool(fe_spec.get("lag_map") is not None),
        "lags": "" if fe_spec.get("lags") is None else ",".join(str(x) for x in fe_spec.get("lags")),
        "rm_windows": "" if not fe_spec.get("rm_windows") else ",".join(str(x) for x in fe_spec.get("rm_windows")),
        "ema_spans": "" if not fe_spec.get("ema_spans") else ",".join(str(x) for x in fe_spec.get("ema_spans")),
        "tsfresh_on": bool(fe_spec.get("tsfresh_path")),
        "fm_on": bool(fe_spec.get("fm_pred_path")),
    }

def _digest_lag_map(lm: Optional[Dict[str, List[int]]]) -> str:
    if not lm:
        return ""
    payload = "|".join(f"{k}:{','.join(map(str, sorted(v)))}" for k, v in sorted(lm.items()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _cache_key(t: int, base_features: List[str], fe_spec: Dict, fs_cfg: FeatureSelectCfg) -> str:
    fe = {
        "lags": tuple(fe_spec.get("lags") or ()),
        "rm": tuple(fe_spec.get("rm_windows") or ()),
        "ema": tuple(fe_spec.get("ema_spans") or ()),
        "pca_stage": (fe_spec.get("pca_stage") or PCA_STAGE_DEFAULT),
        "pca_n": fe_spec.get("pca_n"),
        "pca_var": fe_spec.get("pca_var"),
        "pca_solver": fe_spec.get("pca_solver", "auto"),
        "tsfresh": bool(fe_spec.get("tsfresh_path")),
        "fm": bool(fe_spec.get("fm_pred_path")),
        "lag_map_digest": _digest_lag_map(fe_spec.get("lag_map")),
    }
    fs = {
        "mode": getattr(fs_cfg, "mode", None),
        "topk": int(getattr(fs_cfg, "topk", 0) or 0),
        "min_abs_corr": float(getattr(fs_cfg, "min_abs_corr", 0.0) or 0.0),
        "variance_thresh": float(getattr(fs_cfg, "variance_thresh", 0.0) or 0.0),
        "manual_cols_digest": hashlib.md5(
            ("|".join(sorted(getattr(fs_cfg, "manual_cols", []) or []))).encode("utf-8")
        ).hexdigest(),
    }
    bf_d = hashlib.md5(("|".join(map(str, base_features))).encode("utf-8")).hexdigest()
    payload = {"t": int(t), "fe": fe, "fs": fs, "bf": bf_d}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def score_config_for_next_step(
    X: pd.DataFrame, y: pd.Series, t: int,
    base_features: List[str], fe_spec: Dict,
    fs_cfg: FeatureSelectCfg,
    model_name: str, model_params: Dict, metric_fn,
    train_eval_cfg: Optional[TrainEvalCfg] = None,
    design_cache: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]]] = None
) -> Tuple[float, float, Optional[List[str]], Dict]:
    if model_name.lower() in BASELINE_MODELS:
        ytr = y.iloc[:t+1]
        est = build_estimator(model_name, dict(model_params))
        est.fit(None, ytr)
        yhat = float(np.asarray(est.predict([[0]]))[0])
        val = float(metric_fn([y.iloc[t+1]], [yhat]))
        return val, yhat, None, {"used_es": False, "best_iteration": None}

    key = _cache_key(t, base_features, fe_spec, fs_cfg)
    Mtr_fin = ytr2 = Mev_fin = eng_cols = None
    if design_cache is not None and key in design_cache:
        Mtr_fin, ytr2, Mev_fin, eng_cols = design_cache[key]

    if Mtr_fin is None:
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

        if design_cache is not None:
            design_cache[key] = (Mtr_fin, ytr2, Mev_fin, eng_cols)

    est = build_estimator(model_name, dict(model_params))

    used_es = False
    best_iter = None
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
                used_es = True
                best_iter = getattr(est, "best_iteration_", None)
                if best_iter is None:
                    best_iter = getattr(est, "best_iteration", None)
            except Exception:
                est.fit(Mtr_fin, ytr2)
        else:
            est.fit(Mtr_fin, ytr2)
    else:
        est.fit(Mtr_fin, ytr2)

    yhat = float(np.asarray(est.predict(Mev_fin))[0])
    val = float(metric_fn([y.iloc[t+1]], [yhat]))
    fit_info = {"used_es": bool(used_es), "best_iteration": None if best_iter is None else int(best_iter)}
    return val, yhat, eng_cols, fit_info

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
            sc, _, _, _ = score_config_for_next_step(
                X, y, t, base_features, fe_spec, fs_cfg, model_name, hp, metric_fn, train_eval_cfg,
                design_cache=None
            )
            scores.append(sc)
        except Exception:
            continue
    if not scores:
        return float("inf")
    return float(np.mean(scores))

def _is_booster(model_name: str) -> bool:
    return model_name.lower() in {"xgb", "xgboost", "lgbm", "lightgbm"}

def _derive_numeric_bounds(hp_list: List[Dict], keys: List[str]) -> Dict[str, Tuple[float, float, bool]]:
    bounds = {}
    for k in keys:
        vals = [v[k] for v in hp_list if k in v and isinstance(v[k], (int, float))]
        if not vals:
            continue
        lo, hi = float(min(vals)), float(max(vals))
        is_int = all(isinstance(v, int) for v in vals)
        bounds[k] = (lo, hi, is_int)
    return bounds

def _local_bo_refine(
    X: pd.DataFrame, y: pd.Series, base_features: List[str],
    fs_cfg: FeatureSelectCfg, fe_spec: Dict, model_name: str,
    best_hp: Dict, hp_list: List[Dict], metric_fn, train_eval_cfg: Optional[TrainEvalCfg],
    t_start: int, steps: int, step_incr: int, bo_cfg: BOCfg,
    progress: bool, progress_fn: Optional[Callable]
) -> Dict:
    if not _is_booster(model_name):
        return {"score": float("inf"), "model_params": best_hp}
    keys = list(bo_cfg.hp_keys)
    if not keys:
        return {"score": float("inf"), "model_params": best_hp}
    bounds = _derive_numeric_bounds(hp_list, keys)
    if not bounds:
        return {"score": float("inf"), "model_params": best_hp}

    rng = np.random.RandomState(None if bo_cfg.seed is None else int(bo_cfg.seed))
    best_sc = _eval_over_steps(X, y, base_features, fs_cfg, model_name, best_hp, fe_spec, metric_fn, train_eval_cfg, t_start, steps, step_incr)
    best_params = dict(best_hp)

    for i in range(int(bo_cfg.n_iter)):
        prop = dict(best_params)
        for k in keys:
            if k not in bounds:
                continue
            lo, hi, is_int = bounds[k]
            center = float(best_params.get(k, (lo + hi) / 2.0))
            rad = float(bo_cfg.radius)
            span = (hi - lo) * rad if hi > lo else max(1.0, abs(center) * rad)
            low = max(lo, center - span)
            high = min(hi, center + span)
            if is_int:
                low_i = int(np.floor(low)); high_i = int(np.ceil(high))
                val = int(round(center)) if low_i >= high_i else int(rng.randint(low_i, high_i + 1))
            else:
                val = float(rng.uniform(low, high))
            prop[k] = val

        sc = _eval_over_steps(X, y, base_features, fs_cfg, model_name, prop, fe_spec, metric_fn, train_eval_cfg, t_start, steps, step_incr)
        _notify(progress, progress_fn, "bo_eval", {"i": i+1, "score": round(sc, 6)})
        if sc < best_sc:
            best_sc = sc
            best_params = prop

    return {"score": best_sc, "model_params": best_params}

# ----- Inneres, zeitbewusstes CV (Blöcke ≤ t_end) -----

def _make_inner_cv_blocks(t_end_1based: int, block_len: int, n_blocks: int) -> List[Tuple[int,int]]:
    blocks = []
    end = int(t_end_1based)
    for i in range(1, int(n_blocks) + 1):
        b_end = end - (int(n_blocks) - i) * int(block_len)
        b_start = b_end - int(block_len) + 1
        blocks.append((b_start, b_end))
    return blocks

def _eval_block(
    X, y, base_features, fs_cfg, model_name, hp, fe_spec, metric_fn, train_eval_cfg,
    val_start_1based: int, val_end_1based: int
) -> List[float]:
    scores = []
    a = int(val_start_1based); b = int(val_end_1based)
    for s_1based in range(a-1, b):  # s is train-end index (0-based), val is s+1
        try:
            sc, _, _, _ = score_config_for_next_step(
                X, y, s_1based, base_features, fe_spec, fs_cfg, model_name, hp, metric_fn, train_eval_cfg,
                design_cache=None
            )
            scores.append(sc)
        except Exception:
            continue
    return scores

def _eval_over_inner_cv_blocks_one(
    X, y, base_features, fs_cfg, model_name, hp, fe_spec, metric_fn, train_eval_cfg,
    blocks: List[Tuple[int,int]], agg: str = "median"
) -> Tuple[float, List[float]]:
    per_block_stats = []
    for (a, b) in blocks:
        scs = _eval_block(X, y, base_features, fs_cfg, model_name, hp, fe_spec, metric_fn, train_eval_cfg, a, b)
        if not scs:
            per_block_stats.append(float("inf"))
        else:
            per_block_stats.append(float(np.median(scs)) if agg == "median" else float(np.mean(scs)))
    overall = float(np.median(per_block_stats)) if agg == "median" else float(np.mean(per_block_stats))
    return overall, per_block_stats

# ----- Hauptorchestrierung -----

def online_rolling_forecast(
    X: pd.DataFrame, y: pd.Series,
    initial_window: int, step: int, horizon: int,
    fs_cfg: FeatureSelectCfg, fe_cfg: FeEngCfg,
    model_name: str, model_grid: Dict, metric_fn: Callable,
    progress: bool = False, progress_fn: Optional[Callable] = None,
    per_feature_lag_refresh_k: Optional[int] = None,
    train_eval_cfg: Optional[TrainEvalCfg] = None,
    asha_cfg: Optional[AshaCfg] = None,
    bo_cfg: Optional[BOCfg] = None,
    report_csv_path: Optional[str] = None,
    tuning_csv_path: Optional[str] = None,
    predictions_csv_path: Optional[str] = None,
    min_rel_improvement: float = 0.0,
    inner_cv_cfg: Optional[Dict] = None   # {"use_inner_cv":bool,"block_len":int,"n_blocks":int,"aggregate":"median|mean","use_mean_rank":bool}
):
    if horizon != 1:
        raise NotImplementedError("Aktuell nur horizon=1.")

    base_features0 = list(X.columns)
    start_t = initial_window - 1  # 0-based; z.B. 239 für initial_window=240

    Xtr0, ytr0 = X.iloc[:start_t+1, :], y.iloc[:start_t+1]
    fe_candidates_init = _fe_candidates_from_cfg(Xtr0, ytr0, base_features0, fe_cfg)

    hp_list = list(expand_grid(model_grid))
    n_hp = len(hp_list)
    n_fe_init = len(fe_candidates_init)
    init_evals = _count_evals(n_hp, n_fe_init, fe_cfg.optimize_fe_for_all_hp)
    _notify(progress, progress_fn, "init_start", {"n_hp": n_hp, "n_fe": n_fe_init, "expected_evals": init_evals})

    tuning_rows_buffer: List[Dict] = []
    pred_rows_buffer: List[Dict] = []
    def _push_tuning_row(row: Dict, flush: bool = False):
        if tuning_csv_path is None:
            return
        tuning_rows_buffer.append(row)
        if flush or len(tuning_rows_buffer) >= 200:
            append_tuning_rows(tuning_csv_path, tuning_rows_buffer)
            tuning_rows_buffer.clear()
    def _push_pred_row(row: Dict, flush: bool = False):
        if predictions_csv_path is None:
            return
        pred_rows_buffer.append(row)
        if flush or len(pred_rows_buffer) >= 500:
            append_predictions_rows(predictions_csv_path, pred_rows_buffer)
            pred_rows_buffer.clear()

    def _row_common(stage: str, phase: str, t: int, hp: Dict, fe_spec: Dict, score: Optional[float], status: str, err: Optional[str], fit_info: Optional[Dict] = None, used_cols: Optional[List[str]] = None):
        fe_sum = _summarize_fe_spec(fe_spec)
        row = {
            "phase": phase,
            "stage": stage,
            "t": int(t),
            "model": model_name,
            "score": score,
            "status": status,
            "err": (err or "")[:500],
            "hp": json.dumps(hp, sort_keys=True),
            "pca_stage": fe_sum["pca_stage"],
            "pca_n": fe_sum["pca_n"],
            "pca_var": fe_sum["pca_var"],
            "per_feature_lags": fe_sum["per_feature_lags"],
            "lags": fe_sum["lags"],
            "rm_windows": fe_sum["rm_windows"],
            "ema_spans": fe_sum["ema_spans"],
            "tsfresh_on": fe_sum["tsfresh_on"],
            "fm_on": fe_sum["fm_on"],
            "used_es": None if fit_info is None else bool(fit_info.get("used_es", False)),
            "best_iteration": None if fit_info is None else fit_info.get("best_iteration", None),
            "n_used_cols": None if used_cols is None else int(len(used_cols)),
        }
        return row

    # === ASHA (Initialsuche) ===
    def _asha_initial_search():
        rng = np.random.RandomState(None if asha_cfg.seed is None else int(asha_cfg.seed))

        def _sample_pairs(n: int):
            pairs = [(hp, fe) for hp in hp_list for fe in fe_candidates_init]
            if n >= len(pairs):
                return pairs
            idx = rng.choice(len(pairs), size=int(n), replace=False)
            return [pairs[i] for i in idx]

        # Inner-CV Blöcke auf t_end = start_t+1 (1-based)
        use_icv = bool(inner_cv_cfg and inner_cv_cfg.get("use_inner_cv", False))
        icv_block_len = int(inner_cv_cfg.get("block_len", 20)) if use_icv else None
        icv_n_blocks = int(inner_cv_cfg.get("n_blocks", 3)) if use_icv else None
        icv_agg = str(inner_cv_cfg.get("aggregate", "median")) if use_icv else "median"
        icv_mean_rank = bool(inner_cv_cfg.get("use_mean_rank", True)) if use_icv else False
        blocks_all = _make_inner_cv_blocks(start_t + 1, icv_block_len, icv_n_blocks) if use_icv else None

        def _stage(pairs, budget, tag):
            best_rows = []
            if use_icv:
                # budget = Anzahl genutzter Blöcke (1/2/3)
                blocks = blocks_all[:int(budget)]
                for (hp, fe) in pairs:
                    try:
                        sc, per_b = _eval_over_inner_cv_blocks_one(
                            X, y, base_features0, fs_cfg, model_name, hp, fe, metric_fn, train_eval_cfg,
                            blocks=blocks, agg=icv_agg
                        )
                        best_rows.append({"score": sc, "hp": hp, "fe": fe, "per_block": per_b})
                        _push_tuning_row(_row_common(stage=tag, phase="asha", t=start_t, hp=hp, fe_spec=fe, score=sc, status="ok", err=None))
                    except Exception as e:
                        _push_tuning_row(_row_common(stage=tag, phase="asha", t=start_t, hp=hp, fe_spec=fe, score=None, status="failed", err=str(e)))
                        continue
                if icv_mean_rank and best_rows:
                    B = len(blocks)
                    ranks = np.zeros(len(best_rows), dtype=float)
                    for b in range(B):
                        vals = [r["per_block"][b] for r in best_rows]
                        order = np.argsort(vals)
                        rk = np.empty_like(order, dtype=float)
                        rk[order] = np.arange(len(vals)) + 1
                        ranks += rk
                    mean_ranks = ranks / float(B)
                    rows = [{"score": float(mean_ranks[i]), "hp": best_rows[i]["hp"], "fe": best_rows[i]["fe"]} for i in range(len(best_rows))]
                    rows.sort(key=lambda z: z["score"])
                    return rows
                else:
                    rows = [{"score": float(r["score"]), "hp": r["hp"], "fe": r["fe"]} for r in best_rows]
                    rows.sort(key=lambda z: z["score"])
                    return rows
            else:
                # klassische Steps
                rows = []
                for (hp, fe) in pairs:
                    try:
                        sc = _eval_over_steps(X, y, base_features0, fs_cfg, model_name, hp, fe, metric_fn, train_eval_cfg, start_t, int(budget), step)
                        rows.append({"score": sc, "hp": hp, "fe": fe})
                        _push_tuning_row(_row_common(stage=tag, phase="asha", t=start_t, hp=hp, fe_spec=fe, score=sc, status="ok", err=None))
                    except Exception as e:
                        _push_tuning_row(_row_common(stage=tag, phase="asha", t=start_t, hp=hp, fe_spec=fe, score=None, status="failed", err=str(e)))
                        continue
                rows.sort(key=lambda z: z["score"])
                return rows

        # Budget: bei Inner-CV = 1/2/3 Blöcke; sonst = steps_b1/b2/b3
        b1 = (1 if use_icv else int(asha_cfg.steps_b1))
        b2 = (2 if use_icv else int(asha_cfg.steps_b2))
        b3 = (3 if use_icv else int(asha_cfg.steps_b3))

        res1 = _stage(_sample_pairs(int(asha_cfg.n_b1)), b1, "B1")
        k1 = max(1, int(len(res1) * float(asha_cfg.promote_frac_1)))
        res2 = _stage([(r["hp"], r["fe"]) for r in res1[:k1]], b2, "B2")
        k2 = max(1, int(len(res2) * float(asha_cfg.promote_frac_2)))
        res3 = _stage([(r["hp"], r["fe"]) for r in res2[:k2]], b3, "B3")
        top = res3[0]
        return {"t": start_t, "fe_spec": top["fe"], "model_params": top["hp"], "score": top["score"], "yhat": None}

    if asha_cfg is not None and asha_cfg.use_asha:
        best_init = _asha_initial_search()
        _notify(progress, progress_fn, "init_done_asha", {"best_score": round(best_init["score"], 6)})
        _push_tuning_row({"phase":"asha","stage":"final_pick","t":int(start_t),"model":model_name,"score":best_init["score"],"status":"ok","err":"","hp":json.dumps(best_init["model_params"],sort_keys=True),**_summarize_fe_spec(best_init["fe_spec"]), "used_es":None,"best_iteration":None,"n_used_cols":None}, flush=True)
    else:
        design_cache_init: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]] = {}
        best_init, evals_done = None, 0
        for i_hp, mp in enumerate(hp_list, start=1):
            for i_fe, fe_spec in enumerate(fe_candidates_init, start=1):
                try:
                    score, yhat, used_cols, fit_info = score_config_for_next_step(
                        X, y, start_t, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn, train_eval_cfg,
                        design_cache=design_cache_init
                    )
                    _push_tuning_row(_row_common(stage="full", phase="init", t=start_t, hp=mp, fe_spec=fe_spec, score=score, status="ok", err=None, fit_info=fit_info, used_cols=used_cols))
                except Exception as e:
                    _push_tuning_row(_row_common(stage="full", phase="init", t=start_t, hp=mp, fe_spec=fe_spec, score=None, status="failed", err=str(e)))
                    continue
                evals_done += 1
                _notify(progress, progress_fn, "init_eval", {"done": evals_done, "total": init_evals, "hp_idx": i_hp, "fe_idx": i_fe, "score": round(score, 6)})
                cand = {"t": start_t, "fe_spec": fe_spec, "model_params": mp, "score": score, "yhat": yhat, "fit_info": fit_info}
                if (best_init is None) or (score < best_init["score"]):
                    best_init = cand
        _notify(progress, progress_fn, "init_done", {"best_score": round(best_init["score"], 6)})
        _push_tuning_row({"phase":"init","stage":"final_pick","t":int(start_t),"model":model_name,"score":best_init["score"],"status":"ok","err":"","hp":json.dumps(best_init["model_params"],sort_keys=True),**_summarize_fe_spec(best_init["fe_spec"]), "used_es":None,"best_iteration":None,"n_used_cols":None}, flush=True)

    # === BO optional ===
    if bo_cfg is not None and bo_cfg.use_bo and _is_booster(model_name):
        fe_for_bo = best_init["fe_spec"]
        hp_center = dict(best_init["model_params"])
        bo_res = _local_bo_refine(
            X, y, base_features0, fs_cfg, fe_for_bo, model_name,
            hp_center, hp_list, metric_fn, train_eval_cfg,
            start_t, int(bo_cfg.steps), step, bo_cfg,
            progress, progress_fn
        )
        _push_tuning_row(_row_common(stage="bo", phase="init", t=start_t, hp=bo_res["model_params"], fe_spec=fe_for_bo, score=bo_res["score"], status="ok", err=None), flush=True)
        if bo_res["score"] < best_init["score"]:
            best_init["model_params"] = bo_res["model_params"]
            best_init["score"] = bo_res["score"]
            _notify(progress, progress_fn, "bo_improved", {"best_score": round(best_init["score"], 6)})

    preds, truths = [], []
    prev_best = best_init
    cached_lag_map: Optional[dict] = None

    for i_step, t0 in enumerate(range(start_t, len(y)-1, step)):
        t_pred = t0 + horizon

        # predict mit vorheriger Best-Config
        used_cols = None
        used_fit_info = {}
        try:
            score_used, yhat, used_cols, used_fit_info = score_config_for_next_step(
                X, y, t0, base_features0, prev_best["fe_spec"], fs_cfg, model_name, prev_best["model_params"], metric_fn, train_eval_cfg,
                design_cache={}  # frisch
            )
            preds.append((y.index[t_pred], yhat))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))
            _notify(progress, progress_fn, "step_predict", {"t": int(t0), "yhat": round(yhat, 6)})
            _push_pred_row({"time": y.index[t_pred], "model": model_name, "y_true": float(y.iloc[t_pred]), "y_hat": float(yhat)})
        except Exception as e:
            _notify(progress, progress_fn, "step_predict_error", {"t": int(t0), "error": str(e)})
            preds.append((y.index[t_pred], np.nan))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))
            _push_pred_row({"time": y.index[t_pred], "model": model_name, "y_true": float(y.iloc[t_pred]), "y_hat": float("nan")})
            score_used = np.nan

        # FE-Kandidaten (Lag-Map Cache)
        use_existing = (
            fe_cfg.per_feature_lags
            and cached_lag_map is not None
            and (per_feature_lag_refresh_k is not None and per_feature_lag_refresh_k > 0)
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

        # Suche nächste Best-Config
        _notify(progress, progress_fn, "step_search_start", {"t": int(t0)})
        best_now = None

        # Inner-CV Blöcke bis t0 (1-based = t0+1)
        use_icv = bool(inner_cv_cfg and inner_cv_cfg.get("use_inner_cv", False))
        if use_icv:
            icv_block_len = int(inner_cv_cfg.get("block_len", 20))
            icv_n_blocks = int(inner_cv_cfg.get("n_blocks", 3))
            icv_agg = str(inner_cv_cfg.get("aggregate", "median"))
            icv_mean_rank = bool(inner_cv_cfg.get("use_mean_rank", True))
            blocks_t = _make_inner_cv_blocks(t0 + 1, icv_block_len, icv_n_blocks)

        design_cache_t: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]] = {}
        rows_tmp = []
        for mp in hp_list:
            fe_list = fe_candidates if fe_cfg.optimize_fe_for_all_hp else [prev_best["fe_spec"]]
            for fe_spec in fe_list:
                try:
                    if use_icv:
                        sc, per_b = _eval_over_inner_cv_blocks_one(
                            X, y, base_features0, fs_cfg, model_name, mp, fe_spec, metric_fn, train_eval_cfg,
                            blocks=blocks_t, agg=icv_agg
                        )
                        rows_tmp.append({"score": sc, "hp": mp, "fe": fe_spec, "per_block": per_b})
                        _push_tuning_row(_row_common(stage="search", phase="step", t=t0, hp=mp, fe_spec=fe_spec, score=sc, status="ok", err=None))
                    else:
                        sc, yh, cand_used_cols, fit_info = score_config_for_next_step(
                            X, y, t0, base_features0, fe_spec, fs_cfg, model_name, mp, metric_fn, train_eval_cfg,
                            design_cache=design_cache_t
                        )
                        rows_tmp.append({"score": sc, "hp": mp, "fe": fe_spec})
                        _push_tuning_row(_row_common(stage="search", phase="step", t=t0, hp=mp, fe_spec=fe_spec, score=sc, status="ok", err=None, fit_info=fit_info, used_cols=cand_used_cols))
                except Exception as e:
                    _push_tuning_row(_row_common(stage="search", phase="step", t=t0, hp=mp, fe_spec=fe_spec, score=None, status="failed", err=str(e)))
                    continue

        if not rows_tmp:
            raise RuntimeError(f"No valid candidate at step t={t0}")

        if use_icv and icv_mean_rank:
            # mean rank über Blöcke
            B = len(blocks_t)
            ranks = np.zeros(len(rows_tmp), dtype=float)
            for b in range(B):
                vals = [r["per_block"][b] for r in rows_tmp]
                order = np.argsort(vals)
                rk = np.empty_like(order, dtype=float)
                rk[order] = np.arange(len(vals)) + 1
                ranks += rk
            mean_ranks = ranks / float(B)
            best_idx = int(np.argmin(mean_ranks))
            best_now = {"t": t0, "fe_spec": rows_tmp[best_idx]["fe"], "model_params": rows_tmp[best_idx]["hp"], "score": float(mean_ranks[best_idx]), "yhat": None, "fit_info": {}}
        else:
            rows_tmp.sort(key=lambda r: r["score"])
            best_now = {"t": t0, "fe_spec": rows_tmp[0]["fe"], "model_params": rows_tmp[0]["hp"], "score": float(rows_tmp[0]["score"]), "yhat": None, "fit_info": {}}

        _notify(progress, progress_fn, "step_done", {"best_score": round(best_now["score"], 6)})

        # Stop-Kriterium (optional)
        keep_prev = False
        try:
            if min_rel_improvement > 0.0 and np.isfinite(score_used) and score_used > 0:
                rel_impr = (float(score_used) - float(best_now["score"])) / float(score_used)
                if rel_impr < float(min_rel_improvement):
                    keep_prev = True
        except Exception:
            keep_prev = False

        if keep_prev:
            _push_tuning_row({"phase":"step","stage":"no_change","t":int(t0),"model":model_name,"score":score_used,"status":"ok","err":"","hp":json.dumps(prev_best["model_params"],sort_keys=True),**_summarize_fe_spec(prev_best["fe_spec"]), "used_es":None,"best_iteration":None,"n_used_cols":None}, flush=True)
        else:
            _push_tuning_row({"phase":"step","stage":"final_pick","t":int(t0),"model":model_name,"score":best_now["score"],"status":"ok","err":"","hp":json.dumps(best_now["model_params"],sort_keys=True),**_summarize_fe_spec(best_now["fe_spec"]), "used_es":None,"best_iteration":None,"n_used_cols":None}, flush=True)
            prev_best = best_now

    preds = pd.Series({ts: val for ts, val in preds}).sort_index()
    truths = pd.Series({ts: val for ts, val in truths}).sort_index()
    _notify(progress, progress_fn, "done", {"n_preds": len(preds)})

    if report_csv_path is not None and len(preds) > 0:
        preds_eval = preds.iloc[1:] if len(preds) > 1 else preds.iloc[:0]
        truths_eval = truths.loc[preds_eval.index]
        y_true = truths_eval.values.astype(float)
        y_hat = preds_eval.values.astype(float)
        final_rmse = _rmse(y_true, y_hat)
        final_mae = _mae(y_true, y_hat)
        row = {
            "model": model_name,
            "final_rmse": final_rmse,
            "final_mae": final_mae,
            "n_preds": int(len(preds)),
        }
        append_summary_row(report_csv_path, row)

    if predictions_csv_path is not None and pred_rows_buffer:
        append_predictions_rows(predictions_csv_path, pred_rows_buffer)
        pred_rows_buffer.clear()

    cfgdf = pd.DataFrame(index=preds.index)
    return preds, truths, cfgdf
