# src/tuning.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional, TypedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import median_abs_deviation

# --- Absolute Importe ---
from src.config import GlobalConfig, outputs_for_model
from src.io_timesplits import stageA_blocks, stageB_months, append_csv
from src.features import (
    select_lags_per_feature, build_engineered_matrix, apply_rm3,
    screen_k1, redundancy_reduce_greedy, fit_dr, transform_dr, _DRMap
)
from src.evaluation import rmse
from src.io_timesplits import load_tsfresh, load_chronos, load_ar

# --- NEU: Definition der Target-Block-Typen ---
# Blöcke, die mit ifo-Features komprimiert werden (PCA/PLS)
HIGH_DIM_TARGET_BLOCKS = ["TSFresh"]
# Blöcke, die die DR umgehen und "geschützt" angehängt werden
PROTECTED_TARGET_BLOCKS = ["AR1", "Chronos"]
# ----------------------------------------------------

# ------------------------ Module-weiter Cache für Target-only-Blöcke ------------------------
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
    """
    Hängt (falls aktivierbar) selektiv TSFresh-, Chronos- und AR-Features per Zeitindex an X_base an.
    Gibt auch den chronos_sigma-Wert für den letzten Zeitstempel von X_base zurück.
    """
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


# ------------------------ Hilfen ------------------------
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
    """
    Vereinheitlichte Leselogik: Erst HP, sonst cfg, sonst harmloser Fallback.
    -> Verhindert Stage-A/Stage-B-Drift.
    """
    # HINWEIS: Diese Funktion ist von der 'pca_then_lag'-Version.
    return dict(
        lag_candidates=tuple(model_hp.get("lag_candidates", getattr(cfg, "lag_candidates", (1, 2, 3, 6, 12)))),
        top_k_lags_per_feature=int(model_hp.get("top_k_lags_per_feature", getattr(cfg, "top_k_lags_per_feature", 1))),
        use_rm3=bool(model_hp.get("use_rm3", getattr(cfg, "use_rm3", False))),
        k1_topk=model_hp.get("k1_topk", getattr(cfg, "k1_topk", 200)),  # Kann None sein
        screen_threshold=model_hp.get("screen_threshold", getattr(cfg, "screen_threshold", None)),
        redundancy_method=str(model_hp.get("redundancy_method", getattr(cfg, "redundancy_method", "greedy"))),
        redundancy_param=model_hp.get("redundancy_param", getattr(cfg, "redundancy_param", 0.90)),  # Kann None sein
        dr_method=str(model_hp.get("dr_method", getattr(cfg, "dr_method", "none"))),
        pca_var_target=float(model_hp.get("pca_var_target", getattr(cfg, "pca_var_target", 0.95))),
        pca_kmax=int(model_hp.get("pca_kmax", getattr(cfg, "pca_kmax", 25))),
        pls_components=int(model_hp.get("pls_components", getattr(cfg, "pls_components", 2))),
        target_block_set=model_hp.get("target_block_set"),
        corr_spec=model_hp.get("corr_spec", getattr(cfg, "corr_spec", None)),
        n_features_to_use=int(model_hp.get("n_features_to_use", getattr(cfg, "n_features_to_use", 20))),
        sample_weight_decay=model_hp.get("sample_weight_decay", getattr(cfg, "sample_weight_decay", None)),

        # --- ENTFERNT: Pipeline-Steuerung ---
        # fe_pipeline_mode=str(model_hp.get("fe_pipeline_mode", "lag_then_select")),
        # pca_k_factors=model_hp.get("pca_k_factors"),
    )


def _lag_of(col) -> int:
    try:
        s = str(col)
        if "__lag" not in s:
            return 0
        return int(s.split("__lag")[-1])
    except Exception:
        return 0


# ------------------------
# PIPELINE 1: Originale FE-Pipeline (Gleis 1 & 2)
# ------------------------
def _fit_predict_one_origin_FULL_FE(
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_hp: Dict[str, Any],
        X: pd.DataFrame,  # <-- WICHTIG: Das ist X_ifo (ungelaggt)
        y: pd.Series,
        t_origin: int,
        cfg: GlobalConfig,
        corr_spec
) -> PredictionLog:
    """
    MODIFIZIERT: Führt 'lag_then_select' aus (pca_then_lag entfernt).
    MODIFIZIERT: Lernt f(X_ifo[t+1], Chronos[t]) -> y[t+1]
    """
    I_t = t_origin + 1  # 1-basiert für FE-Helfer
    hp_eff = _hp(cfg, model_hp)

    # <-- MODIFIZIERT: y shiften, damit FE-Funktionen X_t -> y_t lernen
    y_shifted_for_nowcast = y.shift(1)

    # <-- Target-Blöcke aufteilen ---
    all_blocks = hp_eff["target_block_set"] or []
    pre_dr_blocks = [b for b in all_blocks if b in HIGH_DIM_TARGET_BLOCKS]
    post_dr_blocks = [b for b in all_blocks if b in PROTECTED_TARGET_BLOCKS]

    # Chronos-Sigma (Logging)
    chronos_sigma = np.nan
    if "Chronos" in all_blocks:
        # <-- HINWEIS: Holt Chronos[t], was korrekt ist (Prognose für t+1)
        _, chronos_sigma = _augment_with_target_blocks(X.iloc[:I_t], ["Chronos"])

    # Dispersions (Logging)
    disp_t = np.nan

    # Platzhalter für finale Trainingsdaten
    Xb_tr: Optional[np.ndarray] = None
    y_tr: Optional[pd.Series] = None
    Xb_ev: Optional[np.ndarray] = None
    n_sis = 0
    n_red = 0
    n_dr_final = 0
    taus_model = np.array([], dtype=int)

    # =========================================================================
    # --- KEINE PIPELINE-VERZWEIGUNG MEHR ---
    # =========================================================================

    # --- ALTE PIPELINE (B): LAG-THEN-SELECT ---

    # -------------------- 1) Lag-Selektion (train-only bis t) --------------------
    lag_map, _, D, taus_dummy = select_lags_per_feature(
        X=X, y=y_shifted_for_nowcast, I_t=I_t,  # <-- MODIFIZIERT
        L=hp_eff["lag_candidates"], k=hp_eff["top_k_lags_per_feature"],
        corr_spec=corr_spec,
    )

    # -------------------- 2) Feature Engineering (train-only bis t) --------------------
    X_eng = build_engineered_matrix(X, lag_map)
    if hp_eff["use_rm3"]:
        X_eng = apply_rm3(X_eng)

    X_aug_pre_dr, _ = _augment_with_target_blocks(X_eng.iloc[:I_t], pre_dr_blocks)

    max_lag_used = max([_lag_of(c) for c in X_eng.columns] + [0])
    rm_extra = 2 if hp_eff["use_rm3"] else 0
    head_needed = max_lag_used + rm_extra

    # Kandidaten-τ für Screening: 1..t (τ=t eingeschlossen)
    taus_base = np.arange(1, int(I_t), dtype=int)  # <-- MODIFIZIERT
    taus_scr_mask = (taus_base - head_needed >= 0)
    taus_scr = taus_base[taus_scr_mask] if np.any(taus_scr_mask) else (
        taus_base[-1:] if taus_base.size > 0 else np.array([], dtype=int))
    if len(taus_scr) == 0 and I_t > 1:
        taus_scr = np.array([I_t - 1], dtype=int)  # letzter erlaubter τ ist t <-- MODIFIZIERT

    # -------------------- 3) Screening K1 (prewhitened) --------------------
    keep_cols, scores = screen_k1(
        X_eng=X_aug_pre_dr, y=y_shifted_for_nowcast, I_t=I_t, corr_spec=corr_spec, D=D, taus=taus_scr,
        # <-- MODIFIZIERT
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

    # -------------------- 4) Redundanzreduktion --------------------
    if hp_eff["redundancy_method"] == "greedy":
        kept = redundancy_reduce_greedy(X_sel, corr_spec, D, taus_scr, hp_eff["redundancy_param"], scores=scores)
        X_red_train = X_sel.loc[:, kept]
    else:
        kept = keep_cols
        X_red_train = X_sel

    n_red = len(kept)

    # -------------------- 5) Train-Design (τ ∈ [head..t]) --------------------
    X_eng_full = build_engineered_matrix(X, lag_map)
    if hp_eff["use_rm3"]:
        X_eng_full = apply_rm3(X_eng_full)

    X_aug_full_pre_dr, _ = _augment_with_target_blocks(X_eng_full, pre_dr_blocks)
    X_red_full_pre_dr = X_aug_full_pre_dr.loc[:, kept]

    head_needed_final = max([_lag_of(c) for c in X_red_full_pre_dr.columns] + [0])
    taus_base_model = np.arange(1, int(I_t), dtype=int)  # τ=t eingeschlossen <-- MODIFIZIERT
    taus_model_mask = (taus_base_model - head_needed_final >= 0)
    taus_model = taus_base_model[taus_model_mask] if np.any(taus_model_mask) else (
        taus_base_model[-1:] if taus_base_model.size > 0 else np.array([], dtype=int))
    if len(taus_model) == 0 and I_t > 1:
        taus_model = np.array([I_t - 1], dtype=int)  # <-- MODIFIZIERT

    X_tr_pre_dr = X_red_full_pre_dr.iloc[taus_model, :].copy()
    y_tr = y.iloc[taus_model]  # train auf (X_τ, y_τ) <-- MODIFIZIERT (kein shift)

    # Eval-Zeile bei Datum t+1 (Vorhersage y_{t+1})
    x_eval_pre_dr = X_red_full_pre_dr.loc[[y.index[t_origin + 1]], :].copy()  # <-- MODIFIZIERT (Index t+1)

    # -------------------- 6) DR fit (train-only) und anwenden --------------------
    dr_map = fit_dr(
        X_tr_pre_dr,
        method=hp_eff["dr_method"],
        pca_var_target=hp_eff["pca_var_target"],
        pca_kmax=hp_eff["pca_kmax"],
        pls_components=hp_eff["pls_components"],
    )

    if hp_eff["dr_method"] == "pls":
        Xb_tr = transform_dr(dr_map, X_tr_pre_dr, y=y_tr, fit_pls=True)  # <-- MODIFIZIERT (nutze y_tr)
        Xb_ev = transform_dr(dr_map, x_eval_pre_dr, y=None, fit_pls=False)
    else:
        Xb_tr = transform_dr(dr_map, X_tr_pre_dr)
        Xb_ev = transform_dr(dr_map, x_eval_pre_dr)

    # -------------------- 6.5) Geschützte (post-DR) Blöcke holen und anhängen
    if post_dr_blocks:
        X_post_tr, _ = _augment_with_target_blocks(
            pd.DataFrame(index=X_tr_pre_dr.index),
            post_dr_blocks
        )

        # --- KORREKTUR für Hybrid-Timing ---
        # Hole post-DR Blöcke für Evaluation (indiziert an t_origin)
        eval_chronos_index = y.index[[t_origin]]  # <-- NEU: Definiere Index t
        X_post_ev, _ = _augment_with_target_blocks(
            pd.DataFrame(index=eval_chronos_index),  # <-- MODIFIZIERT: Nutze Index t
            post_dr_blocks
        )
        # --- Ende Korrektur ---

        X_post_tr_np = np.nan_to_num(X_post_tr.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_post_ev_np = np.nan_to_num(X_post_ev.values, nan=0.0, posinf=0.0, neginf=0.0)

        Xb_tr = np.hstack([Xb_tr, X_post_tr_np])
        Xb_ev = np.hstack([Xb_ev, X_post_ev_np])

    n_dr_final = Xb_tr.shape[1]

    # =========================================================================
    # --- GEMEINSAMER FIT/PREDICT-TEIL ---
    # =========================================================================

    # -------------------- 7) Modell fitten & Prognose --------------------
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

    # Guardrail (Safety): τ_max muss t sein
    if len(taus_model) > 0:
        assert (int(np.max(
            taus_model))) == I_t - 1, "Guardrail: Max train index (tau) should be t (I_t - 1)."  # <-- MODIFIZIERT

    return {
        "y_pred": y_hat,
        "n_features_sis": n_sis,
        "n_features_redundant": n_red,
        "n_dr_components": n_dr_final,
        "ifo_dispersion_t": disp_t,
        "chronos_sigma_t": chronos_sigma,
    }


# ------------------------
# PIPELINE 2: Dynamische FI-Pipeline (Gleis 3)
# ------------------------
def _fit_predict_one_origin_DYNAMIC_FI(
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_hp: Dict[str, Any],
        X_full_lagged: pd.DataFrame,  # Erwartet X_eng_full_lagged.parquet
        y: pd.Series,
        rolling_imp: pd.DataFrame,  # Erwartet rolling_mean_importance_60m.parquet
        t_origin: int,
        cfg: GlobalConfig
) -> PredictionLog:
    """
    Trainiert und prognostiziert für einen Origin t, basierend auf dynamisch ausgewählten Top-K Features.
    MODIFIZIERT: Trainiert X_t -> y_t, prognostiziert X_{t+1} -> y_{t+1}.
    (HINWEIS: Diese Pipeline verwendet KEINE Target-Blöcke, daher ist keine Hybrid-Logik nötig)
    """
    I_t = t_origin + 1
    hp_eff = _hp(cfg, model_hp)
    n_features = int(hp_eff["n_features_to_use"])

    # 1) Top-K Features für Zielmonat (t+1) bestimmen
    pred_date = y.index[t_origin + 1]
    try:
        importance_scores_for_t = rolling_imp.loc[pred_date]
    except KeyError:
        pos = rolling_imp.index.get_indexer([pred_date], method="pad")[0]
        importance_scores_for_t = rolling_imp.iloc[pos] if pos != -1 else rolling_imp.iloc[-1]

    if importance_scores_for_t.isnull().all():
        top_k_features = X_full_lagged.columns.tolist()[:n_features]
    else:
        top_k_features = importance_scores_for_t.nlargest(n_features).index.tolist()

    # 2) Submatrix auf y.index reindexen (robust ggü. Lücken)
    X_subset = X_full_lagged[top_k_features].reindex(y.index)

    # 3) Head-Trim im Trainingsfenster: first_valid..t (τ=t eingeschlossen)
    X_train_window = X_subset.iloc[:I_t]
    first_valid_date = X_train_window.first_valid_index()
    first_valid_idx_int = 0 if first_valid_date is None else y.index.get_loc(first_valid_date)

    if first_valid_idx_int > t_origin:
        first_valid_idx_int = max(0, t_origin)

    # Train-Range bis inklusiv I_t (→ inkl. τ = t)
    stop = max(first_valid_idx_int, I_t)
    X_tr = X_subset.iloc[first_valid_idx_int:stop]
    y_tr = y.iloc[first_valid_idx_int:stop]
    mask = ~y_tr.isna()
    X_tr, y_tr = X_tr.loc[mask], y_tr.loc[mask]

    # Eval-Zeile (X_t+1) labelbasiert
    x_eval = X_subset.loc[[y.index[t_origin + 1]]]

    # 4) Modell fitten & Prognose
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

    # Guardrail: Stelle sicher, dass letzte Train-τ <= t
    if len(X_tr) > 0:
        last_train_date = X_tr.index[-1]
        assert last_train_date <= y.index[
            t_origin], "Leakage guard (Dynamic-FI): τ > t im Training erkannt."

    return {
        "y_pred": y_hat,
        "n_features_sis": n_features,  # Wir verwenden Top-K als "SIS"
        "n_features_redundant": n_features,  # Kein Redundanz-Filter
        "n_dr_components": n_features,  # Keine DR
        "ifo_dispersion_t": np.nan,  # Nicht berechnet
        "chronos_sigma_t": np.nan  # Nicht verwendet
    }


# ------------------------ Stage A (blockweises Tuning) ------------------------
def run_stageA(
        model_name: str,
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_grid: List[Dict[str, Any]],
        X: pd.DataFrame,  # <-- WICHTIG: Das ist X_ifo (ungelaggt)
        y: pd.Series,
        cfg: GlobalConfig,
        keep_top_k_final: int = 5,
        min_survivors_per_block: int = 2,
        # --- optionale Argumente für Gleis 3 ---
        X_full_lagged: Optional[pd.DataFrame] = None,
        rolling_imp: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Stage A mit block-basiertem Holdout (ASHA-Stil).
    MODIFIZIERT: Lernt X_t -> y_t
    """
    use_dynamic_fi = (X_full_lagged is not None and rolling_imp is not None)
    _progress(f"[Stage A] Using {'DYNAMIC FI (Gleis 3)' if use_dynamic_fi else 'FULL FE (Gleis 1 & 2)'} pipeline.")

    agg_scores: Dict[str, List[float]] = {}
    hp_by_key: Dict[str, Any] = {}

    # <-- MODIFIZIERT: y shiften, damit FE-Funktionen X_t -> y_t lernen
    y_shifted_for_nowcast = y.shift(1)

    rs = _mk_paths(model_name, cfg)
    T = len(y)
    survivors: List[Dict[str, Any]] = list(model_grid)

    for (train_end, oos_start, oos_end, block_id) in stageA_blocks(cfg, T):
        # OOS-Ende auf sicheren Bereich beschränken (max t = T-2, denn wir brauchen y_{t+1})
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

            # X_red_for_oos: Die Matrix, aus der die OOS-Zeilen gepickt werden
            # Xb_tr: Finale numpy-Trainingsmatrix
            # y_tr: Finales pandas-Trainingsziel
            # dr_map: (Optional) gelerntes DR-Mapping
            X_red_for_oos: Optional[pd.DataFrame] = None
            Xb_tr: Optional[np.ndarray] = None
            y_tr: Optional[pd.Series] = None
            dr_map: Optional[_DRMap] = None

            if use_dynamic_fi:
                # --------- PIPELINE 3: DYNAMIC FI (kausal am Trainingsende) ----------
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
                    cands_sorted = importance_scores_for_anchor.sort_values(ascending=False).index
                    top_k_features = [c for c in cands_sorted if c in X_full_lagged.columns][:n_features]
                    if not top_k_features:
                        top_k_features = X_full_lagged.columns.tolist()[:n_features]

                X_subset = X_full_lagged[top_k_features].reindex(y.index)

                X_train_window = X_subset.iloc[:I_t_train]
                first_valid_date = X_train_window.first_valid_index()
                first_valid_idx_int = 0 if first_valid_date is None else y.index.get_loc(first_valid_date)
                if first_valid_idx_int > train_end:  # <-- MODIFIZIERT
                    first_valid_idx_int = max(0, train_end)

                # Train bis τ = train_end (inkl.)
                stop = max(first_valid_idx_int, I_t_train)  # <-- MODIFIZIERT
                X_tr_pd = X_subset.iloc[first_valid_idx_int:stop]
                y_tr = y.iloc[first_valid_idx_int:stop]  # <-- MODIFIZIERT (kein shift)
                mask = ~y_tr.isna()
                X_tr_pd, y_tr = X_tr_pd.loc[mask], y_tr.loc[mask]

                X_red_for_oos = X_subset
                Xb_tr = X_tr_pd.to_numpy(dtype=float)  # Direkt in numpy

            else:
                # --------- PIPELINE 1/2: FULL FE (MODIFIZIERT) ----------
                all_blocks = hp_eff["target_block_set"] or []
                pre_dr_blocks = [b for b in all_blocks if b in HIGH_DIM_TARGET_BLOCKS]
                post_dr_blocks = [b for b in all_blocks if b in PROTECTED_TARGET_BLOCKS]

                # --- (ENTFERNT: 'pca_then_lag' Pipeline) ---

                # --- ALTE PIPELINE (B): LAG-THEN-SELECT ---

                # 1) Lags bis Trainende wählen
                lag_map, _, D, taus_dummy = select_lags_per_feature(
                    X, y=y_shifted_for_nowcast, I_t=I_t_train,  # <-- MODIFIZIERT
                    L=hp_eff["lag_candidates"], k=hp_eff["top_k_lags_per_feature"], corr_spec=hp_eff["corr_spec"]
                )

                # 2) Train-Features bauen (bis t=train_end)
                X_eng = build_engineered_matrix(X, lag_map)
                if hp_eff["use_rm3"]:
                    X_eng = apply_rm3(X_eng)

                X_aug_pre_dr, _ = _augment_with_target_blocks(X_eng.iloc[:I_t_train], pre_dr_blocks)

                max_lag_used = max([_lag_of(c) for c in X_eng.columns] + [0])
                rm_extra = 2 if hp_eff["use_rm3"] else 0
                head_needed = max_lag_used + rm_extra

                # Screening-τ: 1..train_end
                taus_base_A = np.arange(1, int(I_t_train), dtype=int)  # <-- MODIFIZIERT
                taus_scr_mask = (taus_base_A - head_needed >= 0)
                taus_scr = taus_base_A[taus_scr_mask] if np.any(taus_scr_mask) else (
                    taus_base_A[-1:] if taus_base_A.size > 0 else np.array([], dtype=int))
                if len(taus_scr) == 0 and I_t_train > 1:
                    taus_scr = np.array([I_t_train - 1], dtype=int)  # <-- MODIFIZIERT

                # 3) Screening K1
                keep_cols, scores = screen_k1(
                    X_eng=X_aug_pre_dr, y=y_shifted_for_nowcast, I_t=I_t_train, corr_spec=hp_eff["corr_spec"], D=D,
                    taus=taus_scr,  # <-- MODIFIZIERT
                    k1_topk=hp_eff["k1_topk"], threshold=hp_eff["screen_threshold"]
                )
                X_sel = X_aug_pre_dr.loc[:, keep_cols]

                # 4) Redundanz
                if hp_eff["redundancy_method"] == "greedy":
                    kept = redundancy_reduce_greedy(X_sel, hp_eff["corr_spec"], D, taus_scr,
                                                    hp_eff["redundancy_param"],
                                                    scores=scores)
                else:
                    kept = keep_cols

                # 5) Vollständige Matrix für OOS bauen
                X_eng_full = build_engineered_matrix(X, lag_map)
                if hp_eff["use_rm3"]:
                    X_eng_full = apply_rm3(X_eng_full)

                X_aug_full_pre_dr, _ = _augment_with_target_blocks(X_eng_full, pre_dr_blocks)
                X_red_pre_dr = X_aug_full_pre_dr.loc[:, kept]
                X_red_for_oos = X_red_pre_dr  # Speichern für OOS-Schleife

                # 6) Finale Trainingsdaten (bis train_end)
                head_needed_final = max([_lag_of(c) for c in X_red_pre_dr.columns] + [0])
                taus_base_model = np.arange(1, int(I_t_train), dtype=int)  # <-- MODIFIZIERT
                taus_model_mask = (taus_base_model - head_needed_final >= 0)
                taus_model = taus_base_model[taus_model_mask] if np.any(taus_model_mask) else (
                    taus_base_model[-1:] if taus_base_model.size > 0 else np.array([], dtype=int))
                if len(taus_model) == 0 and I_t_train > 1:
                    taus_model = np.array([I_t_train - 1], dtype=int)  # <-- MODIFIZIERT

                X_tr_pre_dr = X_red_pre_dr.iloc[taus_model, :].copy()
                y_tr = y.iloc[taus_model]  # <-- MODIFIZIERT (kein shift)

                # 7) DR auf Trainingsdaten
                dr_map = fit_dr(
                    X_tr_pre_dr, method=hp_eff["dr_method"],
                    pca_var_target=hp_eff["pca_var_target"], pca_kmax=hp_eff["pca_kmax"],
                    pls_components=hp_eff["pls_components"]
                )
                Xb_tr = transform_dr(dr_map, X_tr_pre_dr, y_tr, fit_pls=(hp_eff["dr_method"] == "pls"))

                # 8) Protected Blöcke anhängen
                if post_dr_blocks:
                    X_post_tr, _ = _augment_with_target_blocks(
                        pd.DataFrame(index=X_tr_pre_dr.index),
                        post_dr_blocks
                    )
                    X_post_tr_np = np.nan_to_num(X_post_tr.values, nan=0.0, posinf=0.0, neginf=0.0)
                    Xb_tr = np.hstack([Xb_tr, X_post_tr_np])

                # --- (Ende der 'else: ... FULL FE'-Logik) ---

            # --- Gemeinsamer Fit & Predict-Teil ---
            if Xb_tr is None or y_tr is None or X_red_for_oos is None:
                _progress(f"    WARN: Konnte für Config {i} keine gültigen Trainingsdaten generieren. Überspringe.")
                continue  # Springe zur nächsten HP-Kombination

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

            # ---- Vorhersagen für den gesamten Block mit demselben Fit ----
            for t in range(oos_start - 1, oos_end_eff):  # t läuft bis max T-2
                date_t = y.index[t]  # <-- NEU: Datum t (für Chronos)
                date_t_plus_1 = y.index[t + 1]  # <-- NEU: Datum t+1 (für ifo)

                # <-- NEU: Modifizierte OOS-Datenerstellung
                if use_dynamic_fi:
                    # Dynamic FI: Hole X_ifo[t+1]
                    x_eval_pd = X_red_for_oos.loc[[date_t_plus_1]].copy()
                    Xb_eval = np.nan_to_num(x_eval_pd.values, nan=0.0, posinf=0.0, neginf=0.0)

                else:
                    # Full FE: Hole X_ifo[t+1]
                    x_eval_pre_dr_pd = X_red_for_oos.loc[[date_t_plus_1]].copy()

                    # --- (ENTFERNT: 'pca_then_lag' Hybrid-Timing Korrektur) ---

                    # DR anwenden (falls 'lag_then_select' aktiv war)
                    if dr_map is not None:
                        Xb_eval = transform_dr(dr_map, x_eval_pre_dr_pd, fit_pls=False)
                    else:
                        Xb_eval = np.nan_to_num(x_eval_pre_dr_pd.values, nan=0.0, posinf=0.0, neginf=0.0)

                    # Post-DR Blöcke (nur für 'lag_then_select')
                    if post_dr_blocks:
                        chronos_index_t = y.index[[t]]  # <-- NEU: Index t
                        X_post_ev, _ = _augment_with_target_blocks(
                            pd.DataFrame(index=chronos_index_t),  # <-- MODIFIZIERT: Nutze Index t
                            post_dr_blocks
                        )
                        X_post_ev_np = np.nan_to_num(X_post_ev.values, nan=0.0, posinf=0.0, neginf=0.0)
                        Xb_eval = np.hstack([Xb_eval, X_post_ev_np])

                # --- (Ende der Modifikation) ---

                y_hat = model.predict_one(Xb_eval)
                y_true = float(y.iloc[t + 1])  # <-- Ziel y_{t+1} (unverändert)

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

        # (CSV-Exports & Halving)
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
        # HIER IST DIE 10%-REDUKTION (0.1)
        k_keep = max(min_survivors_per_block, int(np.ceil(len(survivors) * 0.1)))
        k_keep = min(k_keep, len(survivors))
        keep_ids = set(rmse_df_sorted["config_id"].head(k_keep).tolist())
        survivors = [hp for i, hp in enumerate(survivors, start=1) if i in keep_ids]
        _progress(f"[Stage A][Block {block_id}] kept {len(survivors)} configs (floor={min_survivors_per_block}).")

    # (Finale Freeze-Logik)
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


# ------------------------ Stage B (Online-Auswahl) ------------------------
def run_stageB(
        model_name: str,
        model_ctor: Callable[[Dict[str, Any]], Any],
        shortlist: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cfg: GlobalConfig,
        max_months: Optional[int] = None,
        # --- optionale Argumente für Gleis 3 ---
        X_full_lagged: Optional[pd.DataFrame] = None,
        rolling_imp: Optional[pd.DataFrame] = None
) -> None:
    """
    Stage B mit gefrorener Shortlist und Online-Auswahl.
    MODIFIZIERT: Lernt f(X_ifo[t+1], Chronos[t]) -> y[t+1]
    """
    use_dynamic_fi = (X_full_lagged is not None and rolling_imp is not None)
    _progress(f"[Stage B] Using {'DYNAMIC FI (Gleis 3)' if use_dynamic_fi else 'FULL FE (Gleis 1 & 2)'} pipeline.")

    rs = _mk_paths(model_name, cfg)
    T = len(y)

    # Policy-Parameter
    window = int(cfg.policy_window)
    decay = float(cfg.policy_decay)
    gain_min = float(cfg.policy_gain_min)
    cooldown = int(cfg.policy_cooldown)
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

    # Exponentiell abgezinste RMSE über 'window' Fehler mit 'decay'
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

        # Vorhersagen für alle Kandidaten
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

        # Rolling-Fehler aktualisieren
        for i, _, se, _ in yhat_by_cfg:
            rolling_errors[i].append(se)

        # Auswahl nach decayed metric
        wrmse_win = [_wrmse(i) for i in range(len(shortlist))]
        new_idx = int(np.argmin(wrmse_win))
        new_rmse = wrmse_win[new_idx] if new_idx < len(wrmse_win) else float('inf')
        inc_rmse = wrmse_win[active_idx]

        rel_gain = 0.0
        if np.isfinite(inc_rmse) and np.isfinite(new_rmse) and inc_rmse > 0:
            rel_gain = 1.0 - (new_rmse / inc_rmse)

        # Guardrails + Cooldown
        can_switch = (new_idx != active_idx) and (rel_gain >= gain_min)
        if last_switch_t is not None:
            can_switch = can_switch and ((t - last_switch_t) >= cooldown)
        switched = False
        if can_switch:
            active_idx = new_idx
            last_switch_t = t
            switched = True

        # Preds-CSV
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

        # Scores-CSV
        rows2 = []
        for i in range(len(shortlist)):
            rows2.append({
                "t": t, "model": model_name, "config_id": i + 1,
                "wrmse_window": wrmse_win[i], "window_len": len(rolling_errors[i]),
                "active_idx": active_idx + 1,
                "candidate_best_idx": new_idx + 1,
                "gain_vs_incumbent": rel_gain if i == new_idx else 0.0,
                "cooldown": cooldown,
                "cooldown_ok": (last_switch_t is None) or ((t - (last_switch_t or 0)) >= cooldown),
                "switched": switched,
                "selection_mode": selection_mode,
                "policy_window": window, "policy_decay": decay
            })
        append_csv(monthly_scores_path, pd.DataFrame(rows2))

    _progress(f"[Stage B] done → {monthly_dir}")

    # --------- Summary (Overall-RMSE) schreiben ---------
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