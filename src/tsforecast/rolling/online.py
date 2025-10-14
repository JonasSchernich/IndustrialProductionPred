from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import sys, json, hashlib
import numpy as np
import pandas as pd

from ..types import FeatureSelectCfg, FeEngCfg, TrainEvalCfg, AshaCfg, BOCfg
from ..features.selection import select_engineered_features
from ..features.engineering import (
    make_per_feature_lags_by_corr,
    build_engineered_matrix,
    apply_pca_train_transform,
)
from ..models.registry import build_estimator, supports_es
from ..tuning.grid import expand_grid
from ..utils.reporting import append_summary_row, append_tuning_rows, append_predictions_rows
from ..utils.progress import ProgressTracker

PCA_STAGE_DEFAULT = "post"
BASELINE_MODELS = {"mean","avg","average","randomwalk","rw","naive","ar1","ar(1)"}

# --- helpers ---
def _align_after_engineering(M: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    M2 = M.astype(float).ffill()  # nur Vergangenheit
    y2 = y.astype(float)

    # Startpunkt: y vorhanden und wenigstens eine Feature-Spalte vorhanden
    have_any_x = ~M2.isna().all(axis=1)
    mask_start = y2.notna() & have_any_x
    if not mask_start.any():
        return M.iloc[0:0, :], y.iloc[0:0]
    first = int(np.argmax(mask_start.values))
    M2 = M2.iloc[first:, :]
    y2 = y2.iloc[first:]

    # Spalten, die komplett NaN bleiben, entfernen (harmlos, FS filtert ohnehin)
    M2 = M2.dropna(axis=1, how="all")

    return M2, y2


def _rmse(a, b) -> float:
    d = (np.asarray(a)-np.asarray(b)).astype(float)
    return float(np.sqrt(np.mean(d*d)))

def _mae(a, b) -> float:
    d = np.abs((np.asarray(a)-np.asarray(b)).astype(float))
    return float(np.mean(d))

def _summarize_fe_spec(fe_spec: Dict) -> Dict:
    return {
        "pca_stage": fe_spec.get("pca_stage"),
        "pca_n": fe_spec.get("pca_n"),
        "pca_var": fe_spec.get("pca_var"),
        "per_feature_lags": bool(fe_spec.get("lag_map") is not None),
        "lags": "" if fe_spec.get("lags") is None else ",".join(map(str, fe_spec.get("lags"))),
        "rm_windows": "" if not fe_spec.get("rm_windows") else ",".join(map(str, fe_spec.get("rm_windows"))),
        "tsfresh_on": bool(fe_spec.get("tsfresh_path")),
        "fm_on": bool(fe_spec.get("fm_pred_path")),
    }


def _digest_lag_map(lm: Optional[Dict[str, List[int]]]) -> str:
    if not lm: return ""
    payload = "|".join(f"{k}:{','.join(map(str, sorted(v)))}" for k,v in sorted(lm.items()))
    return hashlib.sha1(payload.encode()).hexdigest()

def _cache_key(t, base_features, fe_spec, fs_cfg) -> str:
    fe = {
        "lags": tuple(fe_spec.get("lags") or ()),
        "rm": tuple(fe_spec.get("rm_windows") or ()),
        "pca_stage": (fe_spec.get("pca_stage") or PCA_STAGE_DEFAULT),
        "pca_n": fe_spec.get("pca_n"),
        "pca_var": fe_spec.get("pca_var"),
        "pca_solver": fe_spec.get("pca_solver","auto"),
        "tsfresh": bool(fe_spec.get("tsfresh_path")),
        "fm": bool(fe_spec.get("fm_pred_path")),
        "lag_map_digest": _digest_lag_map(fe_spec.get("lag_map")),
    }
    fs = {
        "mode": getattr(fs_cfg,"mode",None),
        "topk": int(getattr(fs_cfg,"topk",0) or 0),
        "min_abs_corr": float(getattr(fs_cfg,"min_abs_corr",0.0) or 0.0),
        "variance_thresh": float(getattr(fs_cfg,"variance_thresh",0.0) or 0.0),
        "manual_cols_digest": hashlib.md5(("|".join(sorted(getattr(fs_cfg,"manual_cols",[]) or []))).encode()).hexdigest(),
    }
    bf_d = hashlib.md5(("|".join(map(str, base_features))).encode()).hexdigest()
    payload = {"t": int(t), "fe": fe, "fs": fs, "bf": bf_d}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

# --- single-step fit+predict (train-only transforms, causal) ---
def score_config_for_next_step(
    X, y, t, base_features, fe_spec, fs_cfg, model_name, model_params, metric_fn,
    train_eval_cfg: Optional[TrainEvalCfg]=None,
    design_cache: Optional[Dict[str, Tuple[pd.DataFrame,pd.Series,pd.DataFrame,List[str]]]]=None
):
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
        pca_n     = fe_spec.get("pca_n")
        pca_var   = fe_spec.get("pca_var")
        pca_stage = (fe_spec.get("pca_stage") or PCA_STAGE_DEFAULT).lower()

        Xtr = X.iloc[:t+1, :][base_features]
        Xev = X.iloc[:t+2, :][base_features]

        if pca_stage == "pre" and (pca_n is not None or pca_var is not None):
            PCtr, PCev = apply_pca_train_transform(
                Xtr, Xev, pca_n=pca_n, pca_var=pca_var,
                pca_solver=fe_spec.get("pca_solver", "auto")
            )
            pc_cols  = list(PCtr.columns)
            Mtr      = build_engineered_matrix(PCtr, pc_cols, fe_spec)
            Mev_full = build_engineered_matrix(PCev, pc_cols, fe_spec).iloc[[-1], :]
            if Mtr.shape[1] == 0:
                raise ValueError("no cols")
            Mtr2, ytr2 = _align_after_engineering(Mtr, y.iloc[:t+1])
            eng_cols   = select_engineered_features(Mtr2, ytr2, fs_cfg)
            if len(eng_cols) == 0:
                raise ValueError("no selected cols")

            cols = [c for c in eng_cols if pd.notna(Mev_full[c].iloc[-1])]
            if not cols:
                raise ValueError("eval row has NaNs for all selected cols (pre-PCA)")

            Mtr_sel = Mtr2[cols].dropna(axis=0)
            ytr3    = ytr2.loc[Mtr_sel.index]
            if Mtr_sel.shape[0] < 10:
                raise ValueError("too few complete rows after dropna (pre-PCA)")

            Mev_sel = Mev_full[cols]
            ytr2    = ytr3
            Mtr_fin, Mev_fin = Mtr_sel, Mev_sel

        else:
            Mtr      = build_engineered_matrix(Xtr, base_features, fe_spec)
            Mev_full = build_engineered_matrix(Xev, base_features, fe_spec).iloc[[-1], :]
            if Mtr.shape[1] == 0:
                raise ValueError("no cols")
            Mtr2, ytr2 = _align_after_engineering(Mtr, y.iloc[:t+1])
            eng_cols   = select_engineered_features(Mtr2, ytr2, fs_cfg)
            if len(eng_cols) == 0:
                raise ValueError("no selected cols")

            cols = [c for c in eng_cols if pd.notna(Mev_full[c].iloc[-1])]
            if not cols:
                raise ValueError("eval row has NaNs for all selected cols (post-PCA)")

            Mtr_sel = Mtr2[cols].dropna(axis=0)
            ytr3    = ytr2.loc[Mtr_sel.index]
            if Mtr_sel.shape[0] < 10:
                raise ValueError("too few complete rows after dropna (post-PCA)")

            Mev_sel = Mev_full[cols]
            ytr2    = ytr3

            if (pca_n is not None) or (pca_var is not None):
                Mtr_fin, Mev_fin = apply_pca_train_transform(
                    Mtr_sel, Mev_sel,
                    pca_n=pca_n, pca_var=pca_var,
                    pca_solver=fe_spec.get("pca_solver", "auto")
                )
            else:
                Mtr_fin, Mev_fin = Mtr_sel, Mev_sel

        if design_cache is not None:
            design_cache[key] = (Mtr_fin, ytr2, Mev_fin, eng_cols)

    est = build_estimator(model_name, dict(model_params))
    used_es = False
    best_iter = None
    es_rounds = int(getattr(train_eval_cfg, "early_stopping_rounds", 0) or 0)

    if train_eval_cfg and es_rounds > 0 and supports_es(model_name):
        dev_tail = max(0, int(train_eval_cfg.dev_tail))
        if dev_tail > 0 and len(Mtr_fin) > dev_tail:
            Xtr2  = Mtr_fin.iloc[:-dev_tail, :]
            ytr2a = ytr2.iloc[:-dev_tail]
            Xdv   = Mtr_fin.iloc[-dev_tail:, :]
            ydv   = ytr2.iloc[-dev_tail:]
            try:
                est.fit(Xtr2, ytr2a, eval_set=[(Xdv, ydv)],
                        early_stopping_rounds=es_rounds)
                used_es  = True
                best_iter = getattr(est, "best_iteration_", None) or getattr(est, "best_iteration", None)
            except Exception:
                est.fit(Mtr_fin, ytr2)
        else:
            est.fit(Mtr_fin, ytr2)
    else:
        est.fit(Mtr_fin, ytr2)

    yhat = float(np.asarray(est.predict(Mev_fin))[0])
    val  = float(metric_fn([y.iloc[t+1]], [yhat]))
    return val, yhat, eng_cols, {"used_es": used_es, "best_iteration": None if best_iter is None else int(best_iter)}

# --- FE candidates ---
def _fe_candidates_from_cfg(Xtr, ytr, base_features, fe_cfg, existing_lag_map: Optional[dict]=None):
    specs=[]; extra={k:getattr(fe_cfg,k) for k in ("tsfresh_path","fm_pred_path") if hasattr(fe_cfg,k)}
    if fe_cfg.per_feature_lags:
        lag_map = existing_lag_map or make_per_feature_lags_by_corr(Xtr[base_features], ytr, fe_cfg.per_feature_candidates, fe_cfg.per_feature_topk)
        for rms in fe_cfg.candidate_rm_sets:
            for pca_n, pca_var in fe_cfg.candidate_pca:
                for pca_stage in fe_cfg.pca_stage_options:
                    specs.append({"lag_map": lag_map, "rm_windows": rms,
                                  "pca_n": pca_n, "pca_var": pca_var,
                                  "pca_stage": pca_stage, "pca_solver": getattr(fe_cfg, "pca_solver", "auto"),
                                  **extra})
    else:
        for lset in fe_cfg.candidate_lag_sets:
            for rms in fe_cfg.candidate_rm_sets:
                for pca_n, pca_var in fe_cfg.candidate_pca:
                    for pca_stage in fe_cfg.pca_stage_options:
                        specs.append({"lags": lset, "rm_windows": rms,
                                      "pca_n": pca_n, "pca_var": pca_var,
                                      "pca_stage": pca_stage, "pca_solver": getattr(fe_cfg, "pca_solver", "auto"),
                                      **extra})
    return specs

def _count_evals(n_hp, n_fe, full: bool) -> int:
    return n_hp*n_fe if full else n_hp + (n_fe-1)

def _eval_over_steps(X,y,base_features,fs_cfg,model_name,hp,fe_spec,metric_fn,train_eval_cfg,t_start,steps,step_incr):
    scores=[]; n=len(y); end_t=min(t_start+steps-1, n-2)
    for t in range(t_start, end_t+1, step_incr):
        try:
            sc,_,_,_ = score_config_for_next_step(X,y,t,base_features,fe_spec,fs_cfg,model_name,hp,metric_fn,train_eval_cfg,design_cache=None)
            scores.append(sc)
        except: continue
    return float(np.mean(scores)) if scores else float("inf")

def _is_booster(model_name: str) -> bool:
    return model_name.lower() in {"xgb","xgboost","lgbm","lightgbm"}

def _derive_numeric_bounds(hp_list: List[Dict], keys: List[str]) -> Dict[str, Tuple[float,float,bool]]:
    bounds={}
    for k in keys:
        vals=[v[k] for v in hp_list if k in v and isinstance(v[k],(int,float))]
        if not vals: continue
        lo,hi=float(min(vals)),float(max(vals)); is_int=all(isinstance(v,int) for v in vals)
        bounds[k]=(lo,hi,is_int)
    return bounds

def _local_bo_refine(X,y,base_features,fs_cfg,fe_spec,model_name,best_hp,hp_list,metric_fn,train_eval_cfg,t_start,steps,step_incr,bo_cfg,progress,progress_fn):
    if not _is_booster(model_name): return {"score": float("inf"), "model_params": best_hp}
    keys=list(bo_cfg.hp_keys)
    if not keys: return {"score": float("inf"), "model_params": best_hp}
    bounds=_derive_numeric_bounds(hp_list,keys)
    if not bounds: return {"score": float("inf"), "model_params": best_hp}
    rng=np.random.RandomState(None if bo_cfg.seed is None else int(bo_cfg.seed))
    best_sc=_eval_over_steps(X,y,base_features,fs_cfg,model_name,best_hp,fe_spec,metric_fn,train_eval_cfg,t_start,steps,step_incr)
    best_params=dict(best_hp)
    tracker = ProgressTracker("BO", total_units=int(bo_cfg.n_iter), print_every=max(1, int(bo_cfg.n_iter)//10 or 1))
    for i in range(int(bo_cfg.n_iter)):
        prop=dict(best_params)
        for k in keys:
            if k not in bounds: continue
            lo,hi,is_int=bounds[k]
            center=float(best_params.get(k,(lo+hi)/2.0))
            rad=float(bo_cfg.radius); span=(hi-lo)*rad if hi>lo else max(1.0,abs(center)*rad)
            low=max(lo, center-span); high=min(hi, center+span)
            if is_int:
                low_i=int(np.floor(low)); high_i=int(np.ceil(high))
                val=int(round(center)) if low_i>=high_i else int(rng.randint(low_i,high_i+1))
            else:
                val=float(rng.uniform(low,high))
            prop[k]=val
        sc=_eval_over_steps(X,y,base_features,fs_cfg,model_name,prop,fe_spec,metric_fn,train_eval_cfg,t_start,steps,step_incr)
        tracker.update(extra={"i": i+1, "score": round(sc,6)})
        if sc<best_sc: best_sc, best_params = sc, prop
    tracker.finish()
    return {"score": best_sc, "model_params": best_params}

# --- inner time-aware CV blocks (≤ t_end) ---
def _make_inner_cv_blocks(t_end_1based: int, block_len: int, n_blocks: int) -> List[Tuple[int,int]]:
    blocks=[]
    end=int(t_end_1based)
    for i in range(1, int(n_blocks)+1):
        b_end=end-(int(n_blocks)-i)*int(block_len)
        b_start=b_end-int(block_len)+1
        blocks.append((b_start,b_end))
    return blocks

def _eval_block(X,y,base_features,fs_cfg,model_name,hp,fe_spec,metric_fn,train_eval_cfg,val_start_1b:int,val_end_1b:int) -> List[float]:
    scores=[]
    a=int(val_start_1b); b=int(val_end_1b)
    for s_1b in range(a-1, b):
        try:
            sc,_,_,_ = score_config_for_next_step(X,y,s_1b,base_features,fe_spec,fs_cfg,model_name,hp,metric_fn,train_eval_cfg,design_cache=None)
            scores.append(sc)
        except: continue
    return scores

def _eval_over_inner_cv_blocks_one(X,y,base_features,fs_cfg,model_name,hp,fe_spec,metric_fn,train_eval_cfg,blocks,agg="median"):
    stats=[]
    for (a,b) in blocks:
        scs=_eval_block(X,y,base_features,fs_cfg,model_name,hp,fe_spec,metric_fn,train_eval_cfg,a,b)
        stats.append(float(np.median(scs)) if scs and agg=="median" else (float(np.mean(scs)) if scs else float("inf")))
    overall=float(np.median(stats)) if (stats and agg=="median") else (float(np.mean(stats)) if stats else float("inf"))
    return overall, stats

# --- window policy scoring (≤ t) ohne Lookahead ---
def _window_policy_score(X,y,base_features,fs_cfg,model_name,hp,fe_spec,metric_fn,train_eval_cfg,t0_0b:int,window_len:int) -> float:
    end_1b = t0_0b
    if end_1b < 2:
        return float("inf")
    a_1b = max(2, end_1b - window_len + 1)
    scs=_eval_block(X,y,base_features,fs_cfg,model_name,hp,fe_spec,metric_fn,train_eval_cfg,a_1b,end_1b)
    return float(np.median(scs)) if scs else float("inf")

# --- temporal ensemble weights ---
def _te_weights(ages: List[int], half_life: float, min_active_w: float) -> List[float]:
    lam = np.log(2.0) / float(max(half_life, 1e-12))
    raw = np.array([np.exp(-lam*max(a,0)) for a in ages], dtype=float)
    if raw.sum() == 0: raw[0] = 1.0
    w = raw / raw.sum()
    if 0 <= min_active_w <= 1 and w[0] < min_active_w:
        rest = max(1e-12, 1.0 - min_active_w)
        w_others = w[1:] / max(1e-12, w[1:].sum())
        w = np.concatenate(([min_active_w], rest * w_others))
    return [float(v) for v in w]

# --- main ---
def online_rolling_forecast(
    X: pd.DataFrame, y: pd.Series,
    initial_window: int, step: int, horizon: int,
    fs_cfg: FeatureSelectCfg, fe_cfg: FeEngCfg,
    model_name: str, model_grid: Dict, metric_fn: Callable,
    progress: bool=False, progress_fn: Optional[Callable]=None,
    per_feature_lag_refresh_k: Optional[int]=None,
    train_eval_cfg: Optional[TrainEvalCfg]=None,
    asha_cfg: Optional[AshaCfg]=None,
    bo_cfg: Optional[BOCfg]=None,
    report_csv_path: Optional[str]=None,
    tuning_csv_path: Optional[str]=None,
    predictions_csv_path: Optional[str]=None,
    min_rel_improvement: float=0.0,
    inner_cv_cfg: Optional[Dict]=None,
    policy_cfg: Optional[Dict]=None
):
    if horizon!=1: raise NotImplementedError("nur horizon=1")

    base_features0=list(X.columns)
    start_t=initial_window-1

    # initial FE candidates
    Xtr0, ytr0 = X.iloc[:start_t+1,:], y.iloc[:start_t+1]
    fe_candidates_init=_fe_candidates_from_cfg(Xtr0,ytr0,base_features0,fe_cfg)

    # HP grid
    hp_list=list(expand_grid(model_grid))
    n_hp=len(hp_list); n_fe_init=len(fe_candidates_init)
    print(f"[init] hp={n_hp} fe={n_fe_init} evals≈{_count_evals(n_hp,n_fe_init,fe_cfg.optimize_fe_for_all_hp)}", flush=True)

    # optional shortlist freeze
    shortlist = (policy_cfg or {}).get("shortlist")
    frozen_pairs = None
    if shortlist:
        frozen_pairs = [(d["hp"], d["fe_spec"]) for d in shortlist if ("hp" in d and "fe_spec" in d)]

    # buffers
    tuning_rows_buffer=[]; pred_rows_buffer=[]
    def _push_tuning_row(row, flush=False):
        if tuning_csv_path is None: return
        tuning_rows_buffer.append(row)
        if flush or len(tuning_rows_buffer)>=200:
            append_tuning_rows(tuning_csv_path, tuning_rows_buffer); tuning_rows_buffer.clear()
    def _push_pred_row(row, flush=False):
        if predictions_csv_path is None: return
        pred_rows_buffer.append(row)
        if flush or len(pred_rows_buffer)>=500:
            append_predictions_rows(predictions_csv_path, pred_rows_buffer); pred_rows_buffer.clear()

    def _row_common(stage, phase, t, hp, fe_spec, score, status, err, fit_info=None, used_cols=None):
        fe_sum = _summarize_fe_spec(fe_spec)
        # defaults
        pca_stage = fe_sum.get("pca_stage", None)
        pca_n = fe_sum.get("pca_n", None)
        pca_var = fe_sum.get("pca_var", None)
        pfl = bool(fe_sum.get("per_feature_lags", False))
        lags = fe_sum.get("lags", ())
        rm_windows = fe_sum.get("rm_windows", ())
        tsfresh_on = bool(fe_sum.get("tsfresh_on", False))
        fm_on = bool(fe_sum.get("fm_on", False))

        return {
            "phase": phase, "stage": stage, "t": int(t), "model": model_name,
            "score": score, "status": status, "err": (err or "")[:500],
            "hp": json.dumps(hp, sort_keys=True),
            "pca_stage": pca_stage, "pca_n": pca_n, "pca_var": pca_var,
            "per_feature_lags": pfl, "lags": lags, "rm_windows": rm_windows,
            "tsfresh_on": tsfresh_on, "fm_on": fm_on,
            "used_es": None if fit_info is None else bool(fit_info.get("used_es", False)),
            "best_iteration": None if fit_info is None else fit_info.get("best_iteration", None),
            "n_used_cols": None if used_cols is None else int(len(used_cols)),
        }

    # --- ASHA initial search ---
    def _asha_initial_search():
        rng = np.random.RandomState(None if asha_cfg.seed is None else int(asha_cfg.seed))

        def _sample_pairs(n: int):
            pairs_all = [(hp, fe) for hp in hp_list for fe in fe_candidates_init]
            if n >= len(pairs_all): return pairs_all
            n_fe = len(fe_candidates_init)
            base = int(n) // n_fe; rem = int(n) % n_fe
            pairs: List[Tuple[Dict, Dict]] = []
            for j, fe in enumerate(fe_candidates_init):
                take = base + (1 if j < rem else 0)
                take = max(0, min(take, len(hp_list)))
                if take == 0: continue
                hp_idx = rng.choice(len(hp_list), size=take, replace=False)
                for i in hp_idx: pairs.append((hp_list[i], fe))
            if len(pairs) > n:
                idx = rng.choice(len(pairs), size=int(n), replace=False)
                pairs = [pairs[i] for i in idx]
            return pairs

        use_icv = bool(inner_cv_cfg and inner_cv_cfg.get("use_inner_cv", False))
        if use_icv:
            bl = int(inner_cv_cfg.get("block_len",20)); nb=int(inner_cv_cfg.get("n_blocks",3))
            agg= str(inner_cv_cfg.get("aggregate","median")); mean_rank=bool(inner_cv_cfg.get("use_mean_rank",True))
            blocks_all=_make_inner_cv_blocks(start_t+1, bl, nb)
        print(f"[ASHA] B1={asha_cfg.n_b1} B2≈{int(max(1, asha_cfg.n_b1*asha_cfg.promote_frac_1))} B3≈{int(max(1, (asha_cfg.n_b1*asha_cfg.promote_frac_1)*asha_cfg.promote_frac_2))} inner_cv={use_icv}", flush=True)

        def _stage(pairs, budget, tag):
            rows=[]
            stage_name = f"ASHA-{tag}"
            tracker = ProgressTracker(stage_name, total_units=len(pairs), print_every=max(1, len(pairs)//20 or 1))
            if use_icv:
                blocks = blocks_all[:int(budget)]
                tmp=[]
                for (hp,fe) in pairs:
                    try:
                        sc, per = _eval_over_inner_cv_blocks_one(X,y,base_features0,fs_cfg,model_name,hp,fe,metric_fn,train_eval_cfg,blocks,agg=agg)
                        tmp.append({"sc":sc,"hp":hp,"fe":fe,"per":per})
                        _push_tuning_row(_row_common(tag,"asha",start_t,hp,fe,sc,"ok",None))
                        tracker.update(extra={"score": round(sc,6)})
                    except Exception as e:
                        _push_tuning_row(_row_common(tag,"asha",start_t,hp,fe,None,"failed",str(e)))
                        tracker.update(extra={"score":"fail"})
                        continue
                if mean_rank and tmp:
                    B=len(blocks); ranks=np.zeros(len(tmp))
                    for b in range(B):
                        vals=[r["per"][b] for r in tmp]; order=np.argsort(vals); rk=np.empty_like(order, dtype=float); rk[order]=np.arange(len(vals))+1; ranks+=rk
                    mean_r=ranks/float(B); rows=[{"score":float(mean_r[i]),"hp":tmp[i]["hp"],"fe":tmp[i]["fe"]} for i in range(len(tmp))]
                else:
                    rows=[{"score":float(r["sc"]),"hp":r["hp"],"fe":r["fe"]} for r in tmp]
            else:
                for (hp,fe) in pairs:
                    try:
                        sc=_eval_over_steps(X,y,base_features0,fs_cfg,model_name,hp,fe,metric_fn,train_eval_cfg,start_t,int(budget),step)
                        rows.append({"score":sc,"hp":hp,"fe":fe})
                        _push_tuning_row(_row_common(tag,"asha",start_t,hp,fe,sc,"ok",None))
                        tracker.update(extra={"score": round(sc,6)})
                    except Exception as e:
                        _push_tuning_row(_row_common(tag,"asha",start_t,hp,fe,None,"failed",str(e)))
                        tracker.update(extra={"score":"fail"})
                        continue
            tracker.finish()
            rows.sort(key=lambda z:z["score"]); return rows

        b1=(1 if inner_cv_cfg and inner_cv_cfg.get("use_inner_cv",False) else int(asha_cfg.steps_b1))
        b2=(2 if inner_cv_cfg and inner_cv_cfg.get("use_inner_cv",False) else int(asha_cfg.steps_b2))
        b3=(3 if inner_cv_cfg and inner_cv_cfg.get("use_inner_cv",False) else int(asha_cfg.steps_b3))
        r1=_stage(_sample_pairs(int(asha_cfg.n_b1)), b1, "B1")
        k1=max(1,int(len(r1)*float(asha_cfg.promote_frac_1)))
        r2=_stage([(r["hp"],r["fe"]) for r in r1[:k1]], b2, "B2")
        k2=max(1,int(len(r2)*float(asha_cfg.promote_frac_2)))
        r3=_stage([(r["hp"],r["fe"]) for r in r2[:k2]], b3, "B3")
        top=r3[0]
        return {"t":start_t,"fe_spec":top["fe"],"model_params":top["hp"],"score":top["score"],"yhat":None}

    # init (shortlist seed oder asha/full)
    if (policy_cfg or {}).get("shortlist"):
        hp0, fe0 = frozen_pairs[0]
        best_init = {"t": start_t, "fe_spec": fe0, "model_params": hp0, "score": float("inf"), "yhat": None}
        print("[init] shortlist seed", flush=True)
    elif asha_cfg is not None and asha_cfg.use_asha:
        best_init=_asha_initial_search()
        print(f"[init] best_score={round(best_init['score'],6)}", flush=True)
        _push_tuning_row({"phase":"asha","stage":"final_pick","t":int(start_t),"model":model_name,"score":best_init["score"],"status":"ok","err":"","hp":json.dumps(best_init["model_params"],sort_keys=True),**_summarize_fe_spec(best_init["fe_spec"]), "used_es":None,"best_iteration":None,"n_used_cols":None}, flush=True)
    else:
        design_cache_init={}
        best_init=None; total=_count_evals(n_hp,n_fe_init,fe_cfg.optimize_fe_for_all_hp)
        tracker_init = ProgressTracker("INIT", total_units=total, print_every=max(1,total//20 or 1))
        for mp in hp_list:
            for fe_spec in fe_candidates_init:
                try:
                    sc,yh,used_cols,fit_info = score_config_for_next_step(X,y,start_t,base_features0,fe_spec,fs_cfg,model_name,mp,metric_fn,train_eval_cfg,design_cache=design_cache_init)
                    _push_tuning_row(_row_common("full","init",start_t,mp,fe_spec,sc,"ok",None,fit_info,used_cols))
                    tracker_init.update(extra={"score": round(sc,6)})
                except Exception as e:
                    _push_tuning_row(_row_common("full","init",start_t,mp,fe_spec,None,"failed",str(e)))
                    tracker_init.update(extra={"score":"fail"})
                    continue
                cand={"t":start_t,"fe_spec":fe_spec,"model_params":mp,"score":sc,"yhat":yh,"fit_info":fit_info}
                if (best_init is None) or (sc<best_init["score"]): best_init=cand
        tracker_init.finish()
        if best_init is None:
            raise RuntimeError(
                "INIT produced no viable config. Check FE/FS or data gaps (try ffill, fs_cfg.mode='none').")
        print(f"[init] best_score={round(best_init['score'],6)}", flush=True)
        _push_tuning_row({"phase":"init","stage":"final_pick","t":int(start_t),"model":model_name,"score":best_init["score"],"status":"ok","err":"","hp":json.dumps(best_init["model_params"],sort_keys=True),**_summarize_fe_spec(best_init["fe_spec"]), "used_es":None,"best_iteration":None,"n_used_cols":None}, flush=True)

    # policy params
    pol = policy_cfg or {}
    pol_mode = pol.get("mode","incumbent")
    pol_win  = int(pol.get("window_len",12))
    pol_k    = int(pol.get("ensemble_topk",3))
    pol_eps  = float(pol.get("eps",1e-6))
    pol_delay = int(pol.get("delay_steps", 0))          # Default: aus
    pol_cooldown = int(pol.get("cooldown_steps", 0))    # Default: aus

    # temporales Ensemble (nur für Einzel-Policies)
    te_enabled = bool(pol.get("use_temporal_ensemble", False))
    te_memory = int(pol.get("te_memory", 3))
    te_half_life = float(pol.get("te_half_life", 3.0))
    te_min_active_w = float(pol.get("te_min_active_weight", 0.2))

    preds=[]; truths=[]
    cached_lag_map=None

    # aktive Auswahl + Aktivierungen
    active_spec = dict(best_init)
    active_ensemble = None
    activation_queue: List[Tuple[int, object]] = []
    last_switch_t = -10**9
    # Historie für temporales Ensemble (neueste zuerst)
    active_history: List[Tuple[Dict, Dict, int]] = [(active_spec["model_params"], active_spec["fe_spec"], start_t)]

    def _schedule_activation(t0: int, payload):
        nonlocal active_spec, active_ensemble, last_switch_t, active_history
        if pol_delay == 0:
            if pol_mode == "window_ensemble" and isinstance(payload, list):
                active_ensemble = payload
            else:
                active_spec = payload
                last_switch_t = t0
                active_history.insert(0, (active_spec["model_params"], active_spec["fe_spec"], t0))
            return
        activation_queue.append((t0 + pol_delay, payload))

    total_steps = (len(y) - 1) - start_t
    roll_tracker = ProgressTracker("ROLL", total_units=total_steps, print_every=max(1, total_steps//50 or 1))

    for i_step, t0 in enumerate(range(start_t, len(y)-1, step)):
        t_pred = t0 + horizon

        # fällige Aktivierungen
        if activation_queue:
            for (t_act, payload) in list(activation_queue):
                if t_act == t0:
                    if pol_mode == "window_ensemble" and isinstance(payload, list):
                        active_ensemble = payload
                    else:
                        active_ensemble = None
                        active_spec = payload
                        last_switch_t = t0
                        active_history.insert(0, (active_spec["model_params"], active_spec["fe_spec"], t0))
                    activation_queue.remove((t_act, payload))

        # FE candidates (optional lag-map refresh)
        fe_candidates = _fe_candidates_from_cfg(
            X.iloc[:t0+1, :], y.iloc[:t0+1], base_features0, fe_cfg,
            existing_lag_map=(
                cached_lag_map if (
                    fe_cfg.per_feature_lags and cached_lag_map is not None
                    and (per_feature_lag_refresh_k and per_feature_lag_refresh_k > 0)
                    and (i_step % per_feature_lag_refresh_k != 0)
                ) else None
            )
        )
        if fe_cfg.per_feature_lags and fe_candidates:
            lm = fe_candidates[0].get("lag_map")
            if lm is not None: cached_lag_map = lm

        # Auswahl basiert auf Vergangenheit (≤ t), Prognose mit aktiver Spec/Ensemble
        if pol_mode == "incumbent":
            try:
                sc_prev = _window_policy_score(
                    X, y, base_features0, fs_cfg, model_name,
                    active_spec["model_params"], active_spec["fe_spec"],
                    metric_fn, train_eval_cfg, t0, pol_win
                )
            except Exception:
                sc_prev = float("inf")

            cand_specs = (frozen_pairs if frozen_pairs is not None else
                          ([(mp, fe) for mp in hp_list for fe in fe_candidates] if fe_cfg.optimize_fe_for_all_hp
                           else [(active_spec["model_params"], active_spec["fe_spec"])]))
            scores = []
            for (mp, fe) in cand_specs:
                try:
                    sc = _window_policy_score(X, y, base_features0, fs_cfg, model_name, mp, fe, metric_fn,
                                              train_eval_cfg, t0, pol_win)
                    scores.append((sc, mp, fe))
                except Exception:
                    pass
            if not scores:
                scores = [(sc_prev, active_spec["model_params"], active_spec["fe_spec"])]
            scores.sort(key=lambda z: z[0])
            best_sc, best_mp, best_fe = scores[0]

            rel_impr = ((sc_prev - best_sc) / sc_prev) if (
                        np.isfinite(sc_prev) and sc_prev > 0 and np.isfinite(best_sc)) else -np.inf
            ok_cooldown = (pol_cooldown <= 0) or (t0 - last_switch_t >= pol_cooldown)
            if rel_impr >= float(min_rel_improvement) and ok_cooldown:
                _schedule_activation(t0,
                                     {"t": t0, "fe_spec": best_fe, "model_params": best_mp, "score": float(best_sc),
                                      "yhat": None, "fit_info": {}})
                _push_tuning_row(_row_common("policy", "schedule_switch", t0, best_mp, best_fe, best_sc, "ok", ""),
                                 flush=True)
            else:
                _push_tuning_row(
                    _row_common("policy", "keep_active", t0, active_spec["model_params"], active_spec["fe_spec"],
                                sc_prev, "ok", ""), flush=True)

            # Prognose: aktives Setup, optional temporales Ensemble
            def _predict_one(mp, fe):
                try:
                    _, yhat_i, _, _ = score_config_for_next_step(
                        X, y, t0, base_features0, fe, fs_cfg, model_name, mp,
                        metric_fn, train_eval_cfg, design_cache={}
                    )
                except Exception:
                    yhat_i = float("nan")
                return yhat_i

            if te_enabled:
                hist = active_history[:max(1, te_memory)]
                ages = [t0 - a_t for (_, _, a_t) in hist]
                ws = _te_weights(ages, half_life=te_half_life, min_active_w=te_min_active_w)
                yhats = []
                for (mp_i, fe_i, _), w_i in zip(hist, ws):
                    yhats.append(_predict_one(mp_i, fe_i) * float(w_i))
                yhat = float(np.sum(yhats))
            else:
                yhat = _predict_one(active_spec["model_params"], active_spec["fe_spec"])

        elif pol_mode == "window_best":
            cand_specs = (frozen_pairs if frozen_pairs is not None else
                          ([(mp, fe) for mp in hp_list for fe in fe_candidates] if fe_cfg.optimize_fe_for_all_hp
                           else [(active_spec["model_params"], active_spec["fe_spec"])]))
            scores = []
            for (mp, fe) in cand_specs:
                try:
                    sc = _window_policy_score(X, y, base_features0, fs_cfg, model_name, mp, fe, metric_fn,
                                              train_eval_cfg, t0, pol_win)
                    scores.append((sc, mp, fe))
                except Exception:
                    pass
            scores = [s for s in scores if np.isfinite(s[0])]
            if not scores: scores = [(float("inf"), active_spec["model_params"], active_spec["fe_spec"])]
            scores.sort(key=lambda z: z[0])
            sc0, mp0, fe0 = scores[0]

            # Guardrail gegen aktive Spec
            try:
                sc_prev = _window_policy_score(
                    X, y, base_features0, fs_cfg, model_name,
                    active_spec["model_params"], active_spec["fe_spec"],
                    metric_fn, train_eval_cfg, t0, pol_win
                )
            except Exception:
                sc_prev = float("inf")
            rel_impr = ((sc_prev - sc0) / sc_prev) if (
                        np.isfinite(sc_prev) and sc_prev > 0 and np.isfinite(sc0)) else -np.inf
            ok_cooldown = (pol_cooldown <= 0) or (t0 - last_switch_t >= pol_cooldown)
            if (rel_impr >= float(min_rel_improvement)) and ok_cooldown:
                _schedule_activation(t0, {"t": t0, "fe_spec": fe0, "model_params": mp0, "score": float(sc0),
                                          "yhat": None, "fit_info": {}})
                _push_tuning_row(_row_common("policy", "schedule_select", t0, mp0, fe0, sc0, "ok", ""), flush=True)
            else:
                _push_tuning_row(
                    _row_common("policy", "keep_active", t0, active_spec["model_params"], active_spec["fe_spec"],
                                sc_prev, "ok", ""), flush=True)

            # Prognose: aktives Setup, optional temporales Ensemble
            def _predict_one(mp, fe):
                try:
                    _, yhat_i, _, _ = score_config_for_next_step(
                        X, y, t0, base_features0, fe, fs_cfg, model_name, mp,
                        metric_fn, train_eval_cfg, design_cache={}
                    )
                except Exception:
                    yhat_i = float("nan")
                return yhat_i

            if te_enabled:
                hist = active_history[:max(1, te_memory)]
                ages = [t0 - a_t for (_, _, a_t) in hist]
                ws = _te_weights(ages, half_life=te_half_life, min_active_w=te_min_active_w)
                yhats = []
                for (mp_i, fe_i, _), w_i in zip(hist, ws):
                    yhats.append(_predict_one(mp_i, fe_i) * float(w_i))
                yhat = float(np.sum(yhats))
            else:
                yhat = _predict_one(active_spec["model_params"], active_spec["fe_spec"])

        elif pol_mode == "window_ensemble":
            cand_specs = (frozen_pairs if frozen_pairs is not None else
                          ([(mp, fe) for mp in hp_list for fe in fe_candidates] if fe_cfg.optimize_fe_for_all_hp
                           else [(active_spec["model_params"], active_spec["fe_spec"])]))
            scores = []
            for (mp, fe) in cand_specs:
                try:
                    sc = _window_policy_score(X, y, base_features0, fs_cfg, model_name, mp, fe, metric_fn,
                                              train_eval_cfg, t0, pol_win)
                    scores.append((sc, mp, fe))
                except Exception:
                    pass
            scores = [s for s in scores if np.isfinite(s[0])]
            if not scores: scores = [(float("inf"), active_spec["model_params"], active_spec["fe_spec"])]
            scores.sort(key=lambda z: z[0])
            top = scores[:max(1, pol_k)]
            w = np.array([1.0 / (s[0] + pol_eps) for s in top], dtype=float)
            s = float(np.sum(w))
            w = (w / s) if np.isfinite(s) and s > 0 else (np.ones_like(w) / len(w))

            # Guardrail gegen aktive Spec
            try:
                sc_prev = _window_policy_score(
                    X, y, base_features0, fs_cfg, model_name,
                    active_spec["model_params"], active_spec["fe_spec"],
                    metric_fn, train_eval_cfg, t0, pol_win
                )
            except Exception:
                sc_prev = float("inf")
            best_sc = top[0][0]
            rel_impr = ((sc_prev - best_sc) / sc_prev) if (
                        np.isfinite(sc_prev) and sc_prev > 0 and np.isfinite(best_sc)) else -np.inf
            ok_cooldown = (pol_cooldown <= 0) or (t0 - last_switch_t >= pol_cooldown)
            if (rel_impr >= float(min_rel_improvement)) and ok_cooldown:
                _schedule_activation(t0, [(mp_i, fe_i, float(w_i)) for (s_i, mp_i, fe_i), w_i in zip(top, w)])
                _push_tuning_row(_row_common("policy", "schedule_ensemble", t0, {}, {}, float(np.sum(w)), "ok", ""),
                                 flush=True)
            else:
                _push_tuning_row(
                    _row_common("policy", "keep_active", t0, active_spec["model_params"], active_spec["fe_spec"],
                                sc_prev, "ok", ""), flush=True)

            # Prognose: aktuelles Ensemble (temporales Ensemble wird hier bewusst NICHT kombiniert)
            yhats = []
            if active_ensemble:
                for (mp_i, fe_i, w_i) in active_ensemble:
                    try:
                        _, yhat_i, _, _ = score_config_for_next_step(
                            X, y, t0, base_features0, fe_i, fs_cfg, model_name, mp_i, metric_fn, train_eval_cfg,
                            design_cache={}
                        )
                    except Exception:
                        yhat_i = float("nan")
                    yhats.append(yhat_i * float(w_i))
                yhat = float(np.sum(yhats))
            else:
                try:
                    _, yhat, _, _ = score_config_for_next_step(
                        X, y, t0, base_features0, active_spec["fe_spec"], fs_cfg, model_name,
                        active_spec["model_params"],
                        metric_fn, train_eval_cfg, design_cache={}
                    )
                except Exception:
                    yhat = float("nan")
        else:
            raise ValueError(f"Unknown policy mode: {pol_mode}")

            preds.append((y.index[t_pred], yhat))
            truths.append((y.index[t_pred], float(y.iloc[t_pred])))
            _push_pred_row(
                {"time": y.index[t_pred], "model": model_name, "y_true": float(y.iloc[t_pred]), "y_hat": float(yhat)})
            roll_tracker.update(
                extra={"t": int(t0), "pred": str(getattr(y.index[t_pred], "date", lambda: y.index[t_pred])())})


    roll_tracker.finish()

    preds=pd.Series({ts:val for ts,val in preds}).sort_index()
    truths=pd.Series({ts:val for ts,val in truths}).sort_index()

    # Final-Metrik > initial_window
    if report_csv_path is not None and len(preds)>0:
        eval_start_ts = y.index[start_t + 1] if (start_t + 1) < len(y.index) else preds.index[0]
        preds_eval = preds.loc[preds.index >= eval_start_ts]
        truths_eval = truths.loc[preds_eval.index]
        final_rmse=_rmse(truths_eval.values, preds_eval.values) if len(preds_eval)>0 else float("nan")
        final_mae=_mae(truths_eval.values, preds_eval.values) if len(preds_eval)>0 else float("nan")
        append_summary_row(report_csv_path, {"model":model_name,"final_rmse":final_rmse,"final_mae":final_mae,"n_preds":int(len(preds_eval))})

    if predictions_csv_path is not None and pred_rows_buffer:
        append_predictions_rows(predictions_csv_path, pred_rows_buffer); pred_rows_buffer.clear()

    cfgdf=pd.DataFrame(index=preds.index)
    return preds, truths, cfgdf
