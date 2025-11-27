from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional, TypedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import median_abs_deviation

from src.config import GlobalConfig, outputs_for_model
from src.io_timesplits import stageA_blocks, stageB_months, append_csv
from src.features import (
    select_lags_per_feature, build_engineered_matrix,
    screen_k1, redundancy_reduce_greedy, fit_dr, transform_dr, _DRMap
)
from src.evaluation import rmse
from src.io_timesplits import load_tsfresh, load_chronos, load_ar

HIGH_DIM_TARGET_BLOCKS = ["TSFresh"]
PROTECTED_TARGET_BLOCKS = ["AR1", "Chronos"]

_TSFRESH_CACHE: Optional[pd.DataFrame] = None
_CHRONOS_CACHE: Optional[pd.DataFrame] = None
_AR_CACHE: Optional[pd.DataFrame] = None


def _ensure_target_blocks_loaded() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    global _TSFRESH_CACHE, _CHRONOS_CACHE, _AR_CACHE
    if _TSFRESH_CACHE is None:
        try:
            _TSFRESH_CACHE = load_tsfresh()
        except Exception:
            _TSFRESH_CACHE = None
    if _CHRONOS_CACHE is None:
        try:
            _CHRONOS_CACHE = load_chronos()
        except Exception:
            _CHRONOS_CACHE = None
    if _AR_CACHE is None:
        try:
            _AR_CACHE = load_ar()
        except Exception:
            _AR_CACHE = None
    return _TSFRESH_CACHE, _CHRONOS_CACHE, _AR_CACHE


def _augment_with_target_blocks(X_base: pd.DataFrame, block_set: Optional[List[str]]) -> Tuple[pd.DataFrame, float]:
    if not block_set:
        return X_base, np.nan

    Z_ts, Z_ch, Z_ar = _ensure_target_blocks_loaded()
    chronos_sigma = np.nan
    pieces = [X_base]

    if "TSFresh" in block_set and Z_ts is not None:
        pieces.append(Z_ts.reindex(X_base.index))
    if "Chronos" in block_set and Z_ch is not None:
        Z_ch_aligned = Z_ch.reindex(X_base.index)
        pieces.append(Z_ch_aligned)
        col_name = "chronos_std"
        if col_name in Z_ch_aligned.columns and not Z_ch_aligned.empty:
            try:
                chronos_sigma = float(Z_ch_aligned[col_name].iloc[-1])
            except (IndexError, TypeError, ValueError):
                chronos_sigma = np.nan
    if "AR1" in block_set and Z_ar is not None:
        pieces.append(Z_ar.reindex(X_base.index))

    if len(pieces) > 1:
        X_aug = pd.concat(pieces, axis=1)
        return X_aug, chronos_sigma
    else:
        return X_base, chronos_sigma


@dataclass
class RunState:
    cfg: GlobalConfig
    model_name: str
    out_stageA: Path
    out_stageB: Path


class PredictionLog(TypedDict):
    y_pred: float
    n_features_sis: int
    n_features_redundant: int
    n_dr_components: int
    ifo_dispersion_t: float
    chronos_sigma_t: float


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _mk_paths(model_name: str, cfg: GlobalConfig) -> RunState:
    outs = outputs_for_model(model_name)
    return RunState(cfg=cfg, model_name=model_name,
                    out_stageA=outs["stageA"], out_stageB=outs["stageB"])


def _hp(cfg: GlobalConfig, model_hp: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        lag_candidates=tuple(model_hp.get("lag_candidates", getattr(cfg, "lag_candidates", (1, 2, 3, 6, 12)))),
        k1_topk=model_hp.get("k1_topk", getattr(cfg, "k1_topk", 200)),
        screen_threshold=model_hp.get("screen_threshold", getattr(cfg, "screen_threshold", None)),
        redundancy_param=model_hp.get("redundancy_param", getattr(cfg, "redundancy_param", 0.90)),
        dr_method=str(model_hp.get("dr_method", getattr(cfg, "dr_method", "none"))),
        pca_var_target=float(model_hp.get("pca_var_target", getattr(cfg, "pca_var_target", 0.95))),
        pca_kmax=int(model_hp.get("pca_kmax", getattr(cfg, "pca_kmax", 25))),
        pls_components=int(model_hp.get("pls_components", getattr(cfg, "pls_components", 2))),
        target_block_set=model_hp.get("target_block_set"),
        corr_spec=model_hp.get("corr_spec", getattr(cfg, "corr_spec", None)),

        # Setup III Param: Top-N Features
        n_features_to_use=int(model_hp.get("n_features_to_use", getattr(cfg, "n_features_to_use", 20))),
        sample_weight_decay=model_hp.get("sample_weight_decay", getattr(cfg, "sample_weight_decay", None)),
    )


def _lag_of(col) -> int:
    try:
        s = str(col)
        if "__lag" not in s:
            return 0
        return int(s.split("__lag")[-1])
    except Exception:
        return 0


def _fit_predict_one_origin_FULL_FE(
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_hp: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        t_origin: int,
        cfg: GlobalConfig,
        corr_spec
) -> PredictionLog:
    I_t = t_origin + 1
    hp_eff = _hp(cfg, model_hp)

    y_shifted_for_nowcast = y.shift(1)

    all_blocks = hp_eff["target_block_set"] or []
    pre_dr_blocks = [b for b in all_blocks if b in HIGH_DIM_TARGET_BLOCKS]
    post_dr_blocks = [b for b in all_blocks if b in PROTECTED_TARGET_BLOCKS]

    chronos_sigma = np.nan
    if "Chronos" in all_blocks:
        _, chronos_sigma = _augment_with_target_blocks(X.iloc[:I_t], ["Chronos"])

    disp_t = np.nan
    Xb_tr: Optional[np.ndarray] = None
    y_tr: Optional[pd.Series] = None
    Xb_ev: Optional[np.ndarray] = None
    n_sis = 0
    n_red = 0
    n_dr_final = 0
    taus_model = np.array([], dtype=int)

    # 1) Lag-Selektion
    lag_map = select_lags_per_feature(X=X, L=hp_eff["lag_candidates"])

    # 2) Feature Engineering
    X_eng = build_engineered_matrix(X, lag_map)
    X_aug_pre_dr, _ = _augment_with_target_blocks(X_eng.iloc[:I_t], pre_dr_blocks)

    max_lag_used = max([_lag_of(c) for c in X_eng.columns] + [0])
    head_needed = max_lag_used

    taus_base = np.arange(1, int(I_t), dtype=int)
    taus_scr_mask = (taus_base - head_needed >= 0)
    taus_scr = taus_base[taus_scr_mask] if np.any(taus_scr_mask) else (
        taus_base[-1:] if taus_base.size > 0 else np.array([], dtype=int))
    if len(taus_scr) == 0 and I_t > 1:
        taus_scr = np.array([I_t - 1], dtype=int)

    # 3) Screening K1
    keep_cols, scores = screen_k1(
        X_eng=X_aug_pre_dr, y=y_shifted_for_nowcast, I_t=I_t, corr_spec=corr_spec,
        taus=taus_scr,
        k1_topk=hp_eff["k1_topk"], threshold=hp_eff["screen_threshold"]
    )
    X_sel = X_aug_pre_dr.loc[:, keep_cols]
    n_sis = len(keep_cols)

    if not X_sel.empty and len(taus_scr) > 0:
        try:
            X_sel_vals_t = X_sel.iloc[taus_scr].to_numpy(dtype=float)
            disp_t = float(median_abs_deviation(np.nanstd(X_sel_vals_t, axis=0), nan_policy="omit"))
        except Exception:
            disp_t = np.nan

    # 4) Redundanzreduktion
    kept = redundancy_reduce_greedy(X_sel, corr_spec, taus_scr, hp_eff["redundancy_param"], scores=scores)

    n_red = len(kept)

    # 5) Train-Design
    X_eng_full = build_engineered_matrix(X, lag_map)

    X_aug_full_pre_dr, _ = _augment_with_target_blocks(X_eng_full, pre_dr_blocks)
    X_red_full_pre_dr = X_aug_full_pre_dr.loc[:, kept]

    head_needed_final = max([_lag_of(c) for c in X_red_full_pre_dr.columns] + [0])
    taus_base_model = np.arange(1, int(I_t), dtype=int)
    taus_model_mask = (taus_base_model - head_needed_final >= 0)
    taus_model = taus_base_model[taus_model_mask] if np.any(taus_model_mask) else (
        taus_base_model[-1:] if taus_base_model.size > 0 else np.array([], dtype=int))
    if len(taus_model) == 0 and I_t > 1:
        taus_model = np.array([I_t - 1], dtype=int)

    X_tr_pre_dr = X_red_full_pre_dr.iloc[taus_model, :].copy()
    y_tr = y.iloc[taus_model]

    x_eval_pre_dr = X_red_full_pre_dr.loc[[y.index[t_origin + 1]], :].copy()

    # 6) DR fit & anwenden
    dr_map = fit_dr(
        X_tr_pre_dr,
        method=hp_eff["dr_method"],
        pca_var_target=hp_eff["pca_var_target"],
        pca_kmax=hp_eff["pca_kmax"],
        pls_components=hp_eff["pls_components"],
    )

    if hp_eff["dr_method"] == "pls":
        Xb_tr = transform_dr(dr_map, X_tr_pre_dr, y=y_tr, fit_pls=True)
        Xb_ev = transform_dr(dr_map, x_eval_pre_dr, y=None, fit_pls=False)
    else:
        Xb_tr = transform_dr(dr_map, X_tr_pre_dr)
        Xb_ev = transform_dr(dr_map, x_eval_pre_dr)

    # 6.5) Geschützte Blöcke
    if post_dr_blocks:
        X_post_tr, _ = _augment_with_target_blocks(pd.DataFrame(index=X_tr_pre_dr.index), post_dr_blocks)
        eval_chronos_index = y.index[[t_origin + 1]]
        X_post_ev, _ = _augment_with_target_blocks(pd.DataFrame(index=eval_chronos_index), post_dr_blocks)

        X_post_tr_np = np.nan_to_num(X_post_tr.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_post_ev_np = np.nan_to_num(X_post_ev.values, nan=0.0, posinf=0.0, neginf=0.0)

        Xb_tr = np.hstack([Xb_tr, X_post_tr_np])
        Xb_ev = np.hstack([Xb_ev, X_post_ev_np])

    n_dr_final = Xb_tr.shape[1]

    # 7) Modell fitten & Prognose
    hp_seeded = dict(model_hp)
    hp_seeded["seed"] = cfg.seed
    model = model_ctor(hp_seeded)

    weight_decay = hp_eff["sample_weight_decay"]
    if weight_decay is not None:
        try:
            n_train = len(y_tr)
            weights = float(weight_decay) ** np.arange(n_train - 1, -1, -1)
            weights = weights / np.mean(weights)
            model.fit(Xb_tr, np.asarray(y_tr).ravel(), sample_weight=weights)
        except TypeError:
            model.fit(Xb_tr, np.asarray(y_tr).ravel())
    else:
        model.fit(Xb_tr, np.asarray(y_tr).ravel())

    y_hat = float(model.predict_one(Xb_ev))

    if len(taus_model) > 0:
        assert (int(np.max(taus_model))) == I_t - 1, "Guardrail: Max train index should be t."

    return {
        "y_pred": y_hat,
        "n_features_sis": n_sis,
        "n_features_redundant": n_red,
        "n_dr_components": n_dr_final,
        "ifo_dispersion_t": disp_t,
        "chronos_sigma_t": chronos_sigma,
    }


def _fit_predict_one_origin_DYNAMIC_FI(
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_hp: Dict[str, Any],
        X_full_lagged: pd.DataFrame,
        y: pd.Series,
        rolling_imp: pd.DataFrame,
        t_origin: int,
        cfg: GlobalConfig
) -> PredictionLog:
    I_t = t_origin + 1
    hp_eff = _hp(cfg, model_hp)

    # NEU: Strikt Top-N Features (z.B. N=20)
    pred_date = y.index[t_origin + 1]
    n_features = int(hp_eff["n_features_to_use"])

    # 1. Importance-Scores für t holen
    try:
        importance_scores_for_t = rolling_imp.loc[pred_date]
    except KeyError:
        pos = rolling_imp.index.get_indexer([pred_date], method="pad")[0]
        importance_scores_for_t = rolling_imp.iloc[pos] if pos != -1 else rolling_imp.iloc[-1]

    if importance_scores_for_t.isnull().all():
        # Fallback
        top_k_features = X_full_lagged.columns.tolist()[:n_features]
    else:
        # Sortieren & Top N nehmen
        top_k_features = importance_scores_for_t.nlargest(n_features).index.tolist()

    X_subset = X_full_lagged[top_k_features].reindex(y.index)

    # 3. Training & Prediction
    X_train_window = X_subset.iloc[:I_t]
    first_valid_date = X_train_window.first_valid_index()
    first_valid_idx_int = 0 if first_valid_date is None else y.index.get_loc(first_valid_date)

    if first_valid_idx_int > t_origin:
        first_valid_idx_int = max(0, t_origin)

    stop = max(first_valid_idx_int, I_t)
    X_tr = X_subset.iloc[first_valid_idx_int:stop]
    y_tr = y.iloc[first_valid_idx_int:stop]
    mask = ~y_tr.isna()
    X_tr, y_tr = X_tr.loc[mask], y_tr.loc[mask]

    x_eval = X_subset.loc[[y.index[t_origin + 1]]]

    hp_seeded = dict(model_hp)
    hp_seeded["seed"] = cfg.seed
    model = model_ctor(hp_seeded)

    weight_decay = hp_eff["sample_weight_decay"]
    if weight_decay is not None:
        try:
            n_train = len(y_tr)
            weights = float(weight_decay) ** np.arange(n_train - 1, -1, -1)
            weights = weights / np.mean(weights)
            model.fit(X_tr, y_tr, sample_weight=weights)
        except TypeError:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)

    y_hat = float(model.predict_one(x_eval))

    return {
        "y_pred": y_hat,
        "n_features_sis": len(top_k_features),
        "n_features_redundant": len(top_k_features),
        "n_dr_components": len(top_k_features),
        "ifo_dispersion_t": np.nan,
        "chronos_sigma_t": np.nan
    }


def run_stageA(
        model_name: str,
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_grid: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cfg: GlobalConfig,
        keep_top_k_final: int = 5,
        min_survivors_per_block: int = 2,
        X_full_lagged: Optional[pd.DataFrame] = None,
        rolling_imp: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Stage A mit block-basiertem Holdout (ASHA-Stil).
    """
    use_dynamic_fi = (X_full_lagged is not None and rolling_imp is not None)
    _progress(f"[Stage A] Using {'DYNAMIC FI (Gleis 3)' if use_dynamic_fi else 'FULL FE (Gleis 1 & 2)'} pipeline.")

    agg_scores: Dict[str, List[float]] = {}
    hp_by_key: Dict[str, Any] = {}

    y_shifted_for_nowcast = y.shift(1)

    rs = _mk_paths(model_name, cfg)
    T = len(y)
    survivors: List[Dict[str, Any]] = list(model_grid)

    for (train_end, oos_start, oos_end, block_id) in stageA_blocks(cfg, T):
        oos_end_eff = min(oos_end, T - 1)
        if oos_end_eff < oos_start:
            break

        _progress(
            f"[Stage A][Block {block_id}] train_end={train_end}, OOS={oos_start}-{oos_end_eff} | configs={len(survivors)}")
        preds_records: List[Dict[str, Any]] = []
        rmse_records: List[Dict[str, Any]] = []

        I_t_train = train_end + 1

        for i, hp in enumerate(survivors, start=1):
            _progress(f"  - Config {i}/{len(survivors)}")
            y_true_block, y_pred_block = [], []
            n_months = (oos_end_eff - oos_start + 1)

            hp_eff = _hp(cfg, hp)
            hp_seeded = dict(hp)
            hp_seeded["seed"] = cfg.seed
            model = model_ctor(hp_seeded)
            weight_decay = hp_eff["sample_weight_decay"]

            X_red_for_oos: Optional[pd.DataFrame] = None
            Xb_tr: Optional[np.ndarray] = None
            y_tr: Optional[pd.Series] = None
            dr_map: Optional[_DRMap] = None

            if use_dynamic_fi:
                # Dynamic FI: Nur subsetting
                n_features = int(hp_eff["n_features_to_use"])
                anchor_date = y.index[train_end]
                try:
                    importance_scores_for_anchor = rolling_imp.loc[:anchor_date].iloc[-1]
                except Exception:
                    pos = rolling_imp.index.get_indexer([anchor_date], method="pad")[0]
                    importance_scores_for_anchor = rolling_imp.iloc[pos] if pos != -1 else rolling_imp.iloc[0]

                if importance_scores_for_anchor.isnull().all():
                    top_k_features = X_full_lagged.columns.tolist()[:n_features]
                else:
                    top_k_features = importance_scores_for_anchor.nlargest(n_features).index.tolist()

                X_subset = X_full_lagged[top_k_features].reindex(y.index)

                X_train_window = X_subset.iloc[:I_t_train]
                first_valid_date = X_train_window.first_valid_index()
                first_valid_idx_int = 0 if first_valid_date is None else y.index.get_loc(first_valid_date)
                if first_valid_idx_int > train_end:
                    first_valid_idx_int = max(0, train_end)

                stop = max(first_valid_idx_int, I_t_train)
                X_tr_pd = X_subset.iloc[first_valid_idx_int:stop]
                y_tr = y.iloc[first_valid_idx_int:stop]
                mask = ~y_tr.isna()
                X_tr_pd, y_tr = X_tr_pd.loc[mask], y_tr.loc[mask]

                X_red_for_oos = X_subset
                Xb_tr = X_tr_pd.to_numpy(dtype=float)

            else:
                all_blocks = hp_eff["target_block_set"] or []
                pre_dr_blocks = [b for b in all_blocks if b in HIGH_DIM_TARGET_BLOCKS]
                post_dr_blocks = [b for b in all_blocks if b in PROTECTED_TARGET_BLOCKS]

                # 1) Lags
                lag_map = select_lags_per_feature(
                    X,
                    L=hp_eff["lag_candidates"]
                )

                # 2) Train-Features
                X_eng = build_engineered_matrix(X, lag_map)

                X_aug_pre_dr, _ = _augment_with_target_blocks(X_eng.iloc[:I_t_train], pre_dr_blocks)

                max_lag_used = max([_lag_of(c) for c in X_eng.columns] + [0])
                head_needed = max_lag_used

                taus_base_A = np.arange(1, int(I_t_train), dtype=int)
                taus_scr_mask = (taus_base_A - head_needed >= 0)
                taus_scr = taus_base_A[taus_scr_mask] if np.any(taus_scr_mask) else (
                    taus_base_A[-1:] if taus_base_A.size > 0 else np.array([], dtype=int))
                if len(taus_scr) == 0 and I_t_train > 1:
                    taus_scr = np.array([I_t_train - 1], dtype=int)

                # 3) Screening K1
                keep_cols, scores = screen_k1(
                    X_eng=X_aug_pre_dr, y=y_shifted_for_nowcast, I_t=I_t_train, corr_spec=hp_eff["corr_spec"],
                    taus=taus_scr,
                    k1_topk=hp_eff["k1_topk"], threshold=hp_eff["screen_threshold"]
                )
                X_sel = X_aug_pre_dr.loc[:, keep_cols]

                # 4) Redundanz
                kept = redundancy_reduce_greedy(X_sel, hp_eff["corr_spec"], taus_scr,
                                                hp_eff["redundancy_param"],
                                                scores=scores)

                # 5) OOS Matrix
                X_eng_full = build_engineered_matrix(X, lag_map)

                X_aug_full_pre_dr, _ = _augment_with_target_blocks(X_eng_full, pre_dr_blocks)
                X_red_pre_dr = X_aug_full_pre_dr.loc[:, kept]
                X_red_for_oos = X_red_pre_dr

                # 6) Finale Train-Daten
                head_needed_final = max([_lag_of(c) for c in X_red_pre_dr.columns] + [0])
                taus_base_model = np.arange(1, int(I_t_train), dtype=int)
                taus_model_mask = (taus_base_model - head_needed_final >= 0)
                taus_model = taus_base_model[taus_model_mask] if np.any(taus_model_mask) else (
                    taus_base_model[-1:] if taus_base_model.size > 0 else np.array([], dtype=int))
                if len(taus_model) == 0 and I_t_train > 1:
                    taus_model = np.array([I_t_train - 1], dtype=int)

                X_tr_pre_dr = X_red_pre_dr.iloc[taus_model, :].copy()
                y_tr = y.iloc[taus_model]

                # 7) DR
                dr_map = fit_dr(
                    X_tr_pre_dr, method=hp_eff["dr_method"],
                    pca_var_target=hp_eff["pca_var_target"], pca_kmax=hp_eff["pca_kmax"],
                    pls_components=hp_eff["pls_components"]
                )
                Xb_tr = transform_dr(dr_map, X_tr_pre_dr, y_tr, fit_pls=(hp_eff["dr_method"] == "pls"))

                # 8) Protected Blöcke
                if post_dr_blocks:
                    X_post_tr, _ = _augment_with_target_blocks(
                        pd.DataFrame(index=X_tr_pre_dr.index),
                        post_dr_blocks
                    )
                    X_post_tr_np = np.nan_to_num(X_post_tr.values, nan=0.0, posinf=0.0, neginf=0.0)
                    Xb_tr = np.hstack([Xb_tr, X_post_tr_np])

            if Xb_tr is None or y_tr is None or X_red_for_oos is None:
                _progress(f"    WARN: Konnte für Config {i} keine gültigen Trainingsdaten generieren. Überspringe.")
                continue

            if weight_decay is not None:
                try:
                    n_train = len(y_tr)
                    weights = float(weight_decay) ** np.arange(n_train - 1, -1, -1)
                    weights = weights / np.mean(weights)
                    model.fit(Xb_tr, y_tr.to_numpy(dtype=float), sample_weight=weights)
                except TypeError:
                    model.fit(Xb_tr, y_tr.to_numpy(dtype=float))
            else:
                model.fit(Xb_tr, y_tr.to_numpy(dtype=float))

            for t in range(oos_start - 1, oos_end_eff):
                date_t = y.index[t]
                date_t_plus_1 = y.index[t + 1]

                if use_dynamic_fi:
                    x_eval_pd = X_red_for_oos.loc[[date_t_plus_1]].copy()
                    Xb_eval = np.nan_to_num(x_eval_pd.values, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    x_eval_pre_dr_pd = X_red_for_oos.loc[[date_t_plus_1]].copy()
                    if dr_map is not None:
                        Xb_eval = transform_dr(dr_map, x_eval_pre_dr_pd, fit_pls=False)
                    else:
                        Xb_eval = np.nan_to_num(x_eval_pre_dr_pd.values, nan=0.0, posinf=0.0, neginf=0.0)

                    if post_dr_blocks:
                        chronos_index_t = y.index[[t + 1]]
                        X_post_ev, _ = _augment_with_target_blocks(pd.DataFrame(index=chronos_index_t), post_dr_blocks)
                        X_post_ev_np = np.nan_to_num(X_post_ev.values, nan=0.0, posinf=0.0, neginf=0.0)
                        Xb_eval = np.hstack([Xb_eval, X_post_ev_np])

                y_hat = model.predict_one(Xb_eval)
                y_true = float(y.iloc[t + 1])

                y_true_block.append(y_true)
                y_pred_block.append(float(y_hat))

                done = len(y_true_block)
                if (done % 5 == 0) or (done == n_months):
                    _progress(
                        f"    · Month {done}/{n_months} processed | running...RMSE={rmse(np.array(y_true_block), np.array(y_pred_block)):.4f}")

                preds_records.append({
                    "block": f"block{block_id}", "t": t,
                    "date_t_plus_1": y.index[t + 1].strftime("%Y-%m-%d"),
                    "y_true": y_true, "y_pred": y_hat,
                    "model": model_name, "config_id": i
                })

            score = rmse(np.array(y_true_block), np.array(y_pred_block))
            rmse_records.append({
                "block": f"block{block_id}", "model": model_name, "config_id": i,
                "rmse": score, "n_oos": len(y_true_block),
                "train_end": train_end, "oos_start": oos_start, "oos_end": oos_end_eff
            })
            key = json.dumps(hp, sort_keys=True)
            agg_scores.setdefault(key, []).append(float(score))
            hp_by_key[key] = hp

        preds_df = pd.DataFrame(preds_records)
        rmse_df = pd.DataFrame(rmse_records)
        preds_path = rs.out_stageA / f"block{block_id}" / "preds.csv"
        rmse_path = rs.out_stageA / f"block{block_id}" / "rmse.csv"
        append_csv(preds_path, preds_df)
        append_csv(rmse_path, rmse_df)

        configs_records = [{"block": f"block{block_id}", "model": model_name,
                            "config_id": i, "config_json": json.dumps(hp)}
                           for i, hp in enumerate(survivors, start=1)]
        configs_df = pd.DataFrame(configs_records)
        configs_path = rs.out_stageA / f"block{block_id}" / "configs.csv"
        append_csv(configs_path, configs_df)

        rmse_df_sorted = rmse_df.sort_values("rmse", ascending=True)
        k_keep = max(min_survivors_per_block, int(np.ceil(len(survivors) * 0.1)))
        k_keep = min(k_keep, len(survivors))
        keep_ids = set(rmse_df_sorted["config_id"].head(k_keep).tolist())
        survivors = [hp for i, hp in enumerate(survivors, start=1) if i in keep_ids]
        _progress(f"[Stage A][Block {block_id}] kept {len(survivors)} configs (floor={min_survivors_per_block}).")

    def _key(hp):
        return json.dumps(hp, sort_keys=True)

    keys_surv = [_key(hp) for hp in survivors]
    med_list: List[Tuple[float, str]] = []
    for k in keys_surv:
        vals = agg_scores.get(k, [])
        med = float(np.median(vals)) if len(vals) else float('inf')
        med_list.append((med, k))
    med_list.sort(key=lambda z: z[0])
    ordered = [hp_by_key[k] for (_, k) in med_list]
    k_final = min(int(keep_top_k_final), len(ordered))
    shortlist = ordered[:k_final]

    (rs.out_stageA / "shortlist.json").write_text(json.dumps(shortlist, indent=2))
    _progress(f"[Stage A] Shortlist saved with {len(shortlist)} configs.")
    return shortlist


def run_stageB(
        model_name: str,
        model_ctor: Callable[[Dict[str, Any]], Any],
        shortlist: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cfg: GlobalConfig,
        max_months: Optional[int] = None,
        X_full_lagged: Optional[pd.DataFrame] = None,
        rolling_imp: Optional[pd.DataFrame] = None
) -> None:
    """
    Stage B mit gefrorener Shortlist und Online-Auswahl.
    """
    use_dynamic_fi = (X_full_lagged is not None and rolling_imp is not None)
    _progress(f"[Stage B] Using {'DYNAMIC FI (Gleis 3)' if use_dynamic_fi else 'FULL FE (Gleis 1 & 2)'} pipeline.")

    rs = _mk_paths(model_name, cfg)
    T = len(y)

    window = int(cfg.policy_window)
    decay = float(cfg.policy_decay)
    selection_mode = str(cfg.selection_mode).lower()

    active_idx = 0
    last_switch_t: Optional[int] = None

    monthly_dir = rs.out_stageB / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    monthly_scores_path = rs.out_stageB / "monthly" / "scores.csv"
    monthly_preds_path = rs.out_stageB / "monthly" / "preds.csv"

    months_iter = [t for t in stageB_months(cfg, T) if (t + 1) < T]
    if max_months is not None:
        months_iter = months_iter[:max_months]

    rolling_errors: Dict[int, List[float]] = {i: [] for i in range(len(shortlist))}

    def _wrmse(i: int) -> float:
        errs = rolling_errors[i][-window:] if window > 0 else rolling_errors[i]
        if len(errs) == 0:
            return float("inf")
        w = np.array([decay ** k for k in range(len(errs) - 1, -1, -1)], dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            return float("inf") if len(errs) > 0 else 0.0
        w /= w_sum
        mse_w = float(np.sum(w * np.array(errs, dtype=float)))
        return float(np.sqrt(mse_w))

    for t in months_iter:
        _progress(f"[Stage B] Month origin t={t} | evaluating {len(shortlist)} configs | active={active_idx + 1}")
        y_truth = float(y.iloc[t + 1])

        yhat_by_cfg: List[Tuple[int, float, float, PredictionLog]] = []
        for i, hp in enumerate(shortlist):
            if use_dynamic_fi:
                result_dict = _fit_predict_one_origin_DYNAMIC_FI(
                    model_ctor=model_ctor, model_hp=hp,
                    X_full_lagged=X_full_lagged, y=y, rolling_imp=rolling_imp,
                    t_origin=t, cfg=cfg
                )
            else:
                result_dict = _fit_predict_one_origin_FULL_FE(
                    model_ctor=model_ctor, model_hp=hp,
                    X=X, y=y, t_origin=t, cfg=cfg,
                    corr_spec=_hp(cfg, hp)["corr_spec"]
                )

            y_hat = result_dict["y_pred"]
            se = (y_truth - y_hat) ** 2
            yhat_by_cfg.append((i, y_hat, se, result_dict))

        for i, _, se, _ in yhat_by_cfg:
            rolling_errors[i].append(se)

        wrmse_win = [_wrmse(i) for i in range(len(shortlist))]
        new_idx = int(np.argmin(wrmse_win))
        new_rmse = wrmse_win[new_idx] if new_idx < len(wrmse_win) else float('inf')
        inc_rmse = wrmse_win[active_idx]

        rel_gain = 0.0
        if np.isfinite(inc_rmse) and np.isfinite(new_rmse) and inc_rmse > 0:
            rel_gain = 1.0 - (new_rmse / inc_rmse)

        can_switch = (new_idx != active_idx)

        switched = False
        if can_switch:
            active_idx = new_idx
            last_switch_t = t
            switched = True

        rows = []
        for i, y_hat, _, result_dict in yhat_by_cfg:
            rows.append({
                "t": t, "date_t_plus_1": y.index[t + 1].strftime("%Y-%m-%d"),
                "y_true": y_truth, "y_pred": y_hat,
                "model": model_name, "config_id": i + 1,
                "is_active": (i == active_idx),
                "wrmse_window": wrmse_win[i],
                "window_len": len(rolling_errors[i]),
                "selection_mode": selection_mode,
                "n_features_sis": result_dict["n_features_sis"],
                "n_features_redundant": result_dict["n_features_redundant"],
                "n_dr_components": result_dict["n_dr_components"],
                "ifo_dispersion_t": result_dict["ifo_dispersion_t"],
                "chronos_sigma_t": result_dict["chronos_sigma_t"]
            })
        append_csv(monthly_preds_path, pd.DataFrame(rows))

        rows2 = []
        for i in range(len(shortlist)):
            rows2.append({
                "t": t, "model": model_name, "config_id": i + 1,
                "wrmse_window": wrmse_win[i], "window_len": len(rolling_errors[i]),
                "active_idx": active_idx + 1,
                "candidate_best_idx": new_idx + 1,
                "gain_vs_incumbent": rel_gain if i == new_idx else 0.0,
                "switched": switched,
                "selection_mode": selection_mode,
                "policy_window": window, "policy_decay": decay
            })
        append_csv(monthly_scores_path, pd.DataFrame(rows2))

    _progress(f"[Stage B] done → {monthly_dir}")

    try:
        dfp = pd.read_csv(monthly_preds_path)
        dfp["se"] = (dfp["y_true"] - dfp["y_pred"]) ** 2
        rmse_by_cfg = (dfp.groupby("config_id")["se"].mean() ** 0.5).reset_index()
        rmse_by_cfg.rename(columns={"se": "rmse_overall"}, inplace=True)
        active = dfp[dfp["is_active"] == True]
        rmse_active = float(((active["y_true"] - active["y_pred"]) ** 2).mean() ** 0.5) if len(active) else np.nan
        summary = rmse_by_cfg.copy()
        summary["model"] = model_name
        summary_path = monthly_dir.parent / "summary" / "summary.csv"
        summary_path.parent.mkdir(exist_ok=True)
        summary.to_csv(summary_path, index=False)
        with open(monthly_dir.parent / "summary" / "summary_active.txt", "w") as f:
            f.write(f"RMSE_active_overall,{rmse_active:.6f}\n")
        _progress(f"[Stage B] summary.csv & summary_active.txt geschrieben.")
    except Exception as e:
        _progress(f"[Stage B] Summary-Schreiben übersprungen: {e}")