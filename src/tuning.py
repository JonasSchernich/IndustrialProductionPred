# src/tuning.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional, TypedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import json

# --- Absolute Importe ---
from src.config import GlobalConfig, outputs_for_model
from src.io_timesplits import stageA_blocks, stageB_months, append_csv
from src.features import (
    select_lags_per_feature, build_engineered_matrix, apply_rm3,
    screen_k1, redundancy_reduce_greedy, fit_dr, transform_dr
)
from src.evaluation import rmse
from src.io_timesplits import load_tsfresh, load_chronos, load_ar

# ------------------------ Module-weiter Cache für Target-only-Blöcke ------------------------

_TSFRESH_CACHE: Optional[pd.DataFrame] = None
_CHRONOS_CACHE: Optional[pd.DataFrame] = None
_AR_CACHE: Optional[pd.DataFrame] = None


def _ensure_target_blocks_loaded() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Lädt TSFresh-/Chronos-/AR-Features einmalig und cached sie.
    Gibt (Z_ts, Z_ch, Z_ar) zurück; kann None sein, wenn Files fehlen.
    """
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


def _augment_with_target_blocks(X_base: pd.DataFrame, block_set: Optional[List[str]]) -> pd.DataFrame:
    """
    Hängt (falls aktivierbar) selektiv TSFresh-, Chronos- und AR-Features per Zeitindex an X_base an.
    """
    if not block_set:
        return X_base

    Z_ts, Z_ch, Z_ar = _ensure_target_blocks_loaded()

    pieces = [X_base]

    if "TSFresh" in block_set and Z_ts is not None:
        pieces.append(Z_ts.reindex(X_base.index))
    if "Chronos" in block_set and Z_ch is not None:
        pieces.append(Z_ch.reindex(X_base.index))
    if "AR1" in block_set and Z_ar is not None:
        pieces.append(Z_ar.reindex(X_base.index))

    X_aug = pd.concat(pieces, axis=1)
    return X_aug


# ------------------------ Hilfen ------------------------

@dataclass
class RunState:
    cfg: GlobalConfig
    model_name: str
    out_stageA: Path
    out_stageB: Path


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _mk_paths(model_name: str, cfg: GlobalConfig) -> RunState:
    outs = outputs_for_model(model_name)
    return RunState(cfg=cfg, model_name=model_name,
                    out_stageA=outs["stageA"], out_stageB=outs["stageB"])


# NEU: TypedDict für den Return-Wert, um das Logging sauber zu halten
class PredictionLog(TypedDict):
    y_pred: float
    n_features_sis: int
    n_features_redundant: int
    n_dr_components: int
    ifo_dispersion_t: float
    chronos_sigma_t: float


# ------------------------ Core: ein Origin schätzen (KORRIGIERT & ERWEITERT) ------------------------

def _fit_predict_one_origin(
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_hp: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        t_origin: int,
        cfg: GlobalConfig,
        corr_spec
) -> PredictionLog:
    """
    Train-only Design bis inkl. Origin t (0-basiert), fit Modell, prognostiziere y_{t+1}.
    Gibt jetzt ein Dict mit Prognose und Analyse-Metriken zurück.
    """
    I_t = t_origin + 1  # 1-basiertes Zählen für die FE-Helfer

    # -------------------- Effektive FE/DR-Parameter pro Konfiguration --------------------
    L_eff = tuple(model_hp.get("lag_candidates", getattr(cfg, "lag_candidates", (1, 2, 3, 6, 12))))
    topk_lags_eff = int(model_hp.get("top_k_lags_per_feature", getattr(cfg, "top_k_lags_per_feature", 1)))
    use_rm3_eff = bool(model_hp.get("use_rm3", getattr(cfg, "use_rm3", False)))
    k1_topk_eff = int(model_hp.get("k1_topk", getattr(cfg, "k1_topk", 200)))
    screen_threshold_eff = model_hp.get("screen_threshold", getattr(cfg, "screen_threshold", None))
    redundancy_method_eff = str(model_hp.get("redundancy_method", getattr(cfg, "redundancy_method", "greedy")))
    redundancy_param_eff = float(model_hp.get("redundancy_param", getattr(cfg, "redundancy_param", 0.90)))
    dr_method_eff = str(model_hp.get("dr_method", getattr(cfg, "dr_method", "none")))
    pca_var_target_eff = float(model_hp.get("pca_var_target", getattr(cfg, "pca_var_target", 0.95)))
    pca_kmax_eff = int(model_hp.get("pca_kmax", getattr(cfg, "pca_kmax", 50)))
    pls_components_eff = int(model_hp.get("pls_components", getattr(cfg, "pls_components", 4)))
    target_block_set_eff = model_hp.get("target_block_set")

    # -------------------- 1) Lag-Selektion --------------------
    lag_map, _, D, taus = select_lags_per_feature(
        X=X, y=y, I_t=I_t, L=L_eff, k=topk_lags_eff,
        corr_spec=corr_spec,
    )

    # -------------------- 2) Feature Engineering --------------------
    X_eng = build_engineered_matrix(X, lag_map)
    if use_rm3_eff:
        X_eng = apply_rm3(X_eng)

    # ---- Target-only Blöcke (Parquet) hier anhängen — VOR Screening/Reduktion/DR ----
    X_aug = _augment_with_target_blocks(X_eng, target_block_set_eff)

    # ---- Head-Trim nur auf Basis der Lags/RM (Target-only sind lag=0) ----
    def _lag_of(col):
        try:
            if "__lag" not in str(col):
                return 0
            return int(str(col).split('__lag')[-1])
        except Exception:
            return 0

    max_lag_used = max([_lag_of(c) for c in X_eng.columns] + [0])
    rm_extra = 2 if use_rm3_eff else 0
    head_needed = max_lag_used + rm_extra

    # taus (von select_lags) startet bei 1
    taus_base = np.arange(1, int(I_t), dtype=int)

    # KORRIGIERT: Head-Trim Off-by-One (Kritik 1)
    taus_scr_mask = (taus_base - head_needed >= 0)
    if np.sum(taus_scr_mask) == 0:
        taus_scr = taus_base[-1:].copy() if taus_base.size > 0 else np.array([], dtype=int)
    else:
        taus_scr = taus_base[taus_scr_mask]

    # -------------------- 3) Screening K1 (prewhitened) --------------------
    keep_cols, scores = screen_k1(
        X_eng=X_aug, y=y, I_t=I_t, corr_spec=corr_spec, D=D, taus=taus_scr,
        k1_topk=k1_topk_eff, threshold=screen_threshold_eff
    )
    X_sel = X_aug.loc[:, keep_cols]

    # --- LOGGING-METRIK 1: n_features_sis ---
    n_sis = len(keep_cols)

    # --- LOGGING-METRIK 2: ifo_dispersion_t ---
    disp_t = np.nan
    if not X_sel.empty:
        try:
            # Berechne Dispersion der *gescreenten* Features (wie in Thesis 5.5)
            X_sel_vals = X_sel.iloc[taus_scr, :].to_numpy(dtype=float)
            # Median der Standardabweichungen jeder Spalte = robustes Maß
            disp_t = float(np.nanmedian(np.nanstd(X_sel_vals, axis=0)))
        except Exception:
            disp_t = np.nan  # Fallback

    # -------------------- 4) Redundanzreduktion --------------------
    if redundancy_method_eff == "greedy":
        kept = redundancy_reduce_greedy(
            X_sel, corr_spec, D, taus_scr, redundancy_param_eff, scores=scores
        )
        X_red = X_sel.loc[:, kept]
    else:
        X_red = X_sel
        kept = keep_cols

    # --- LOGGING-METRIK 3: n_features_redundant ---
    n_red = len(kept)

    # -------------------- 5) Head-Trim / Train-Design --------------------
    head_needed_final = max([_lag_of(c) for c in X_red.columns] + [0])

    # KORRIGIERT: Head-Trim Off-by-One (Kritik 1)
    taus_model_mask = (taus_base - head_needed_final >= 0)
    if np.sum(taus_model_mask) == 0:
        taus_model = taus_base[-1:].copy() if taus_base.size > 0 else np.array([], dtype=int)
    else:
        taus_model = taus_base[taus_model_mask]

    X_tr = X_red.iloc[taus_model, :].copy()
    y_tr = y.shift(-1).iloc[taus_model]

    # Eval-Design: Zeile t_origin
    x_eval = X_red.iloc[[t_origin], :].copy()

    # -------------------- 6) DR fit (train-only) und anwenden --------------------
    dr_map = fit_dr(
        X_tr,
        method=dr_method_eff,
        pca_var_target=pca_var_target_eff,
        pca_kmax=pca_kmax_eff,
        pls_components=pls_components_eff
    )

    # --- LOGGING-METRIK 4: n_dr_components ---
    n_dr = dr_map.n_components_  # Dieser Wert wird in fit_dr korrekt gesetzt

    if dr_method_eff == "pls":
        Xb_tr = transform_dr(dr_map, X_tr, y=y_tr, fit_pls=True)
        Xb_ev = transform_dr(dr_map, x_eval, y=None, fit_pls=False)
    else:
        Xb_tr = transform_dr(dr_map, X_tr)
        Xb_ev = transform_dr(dr_map, x_eval)

    # -------------------- 7) Modell fitten & Prognose --------------------
    model_hp_with_seed = dict(model_hp)
    model_hp_with_seed['seed'] = cfg.seed

    model = model_ctor(model_hp_with_seed)

    # --- Modifikation für Sample Weights ---
    weight_decay = model_hp.get("sample_weight_decay")
    if weight_decay is not None:
        try:
            n_train = len(y_tr)
            weights = float(weight_decay) ** np.arange(n_train - 1, -1, -1)
            weights = weights / np.mean(weights)

            model.fit(np.asarray(Xb_tr), np.asarray(y_tr).ravel(), sample_weight=weights)
        except TypeError:
            model.fit(np.asarray(Xb_tr), np.asarray(y_tr).ravel())
    else:
        model.fit(np.asarray(Xb_tr), np.asarray(y_tr).ravel())
    # --- Ende Modifikation ---

    y_hat = float(model.predict_one(np.asarray(Xb_ev)))

    # --- LOGGING-METRIK 5: chronos_sigma_t ---
    ch_sig = np.nan
    if "Chronos" in (target_block_set_eff or []) and _CHRONOS_CACHE is not None:
        # Annahme: Spalte heißt 'chronos_std' (gemäß Thesis σ(PD))
        col_name = 'chronos_std'
        if col_name in _CHRONOS_CACHE.columns:
            try:
                ch_sig = float(_CHRONOS_CACHE.at[X.index[t_origin], col_name])
            except KeyError:
                ch_sig = np.nan  # Datum t_origin nicht im Cache

    return {
        "y_pred": y_hat,
        "n_features_sis": n_sis,
        "n_features_redundant": n_red,
        "n_dr_components": n_dr,
        "ifo_dispersion_t": disp_t,
        "chronos_sigma_t": ch_sig
    }


# ------------------------ Stage A (KORRIGIERT) ------------------------

def run_stageA(
        model_name: str,
        model_ctor: Callable[[Dict[str, Any]], Any],
        model_grid: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cfg: GlobalConfig,
        keep_top_k_final: int = 5,
        min_survivors_per_block: int = 2,
) -> List[Dict[str, Any]]:
    """
    Stage A mit block-basiertem Holdout (ASHA-Stil).
    """
    agg_scores: Dict[str, List[float]] = {}
    hp_by_key: Dict[str, Any] = {}

    rs = _mk_paths(model_name, cfg)
    T = len(y)
    survivors: List[Dict[str, Any]] = list(model_grid)

    for (train_end, oos_start, oos_end, block_id) in stageA_blocks(cfg, T):
        oos_end_eff = min(oos_end, T - 2)  # T-2, da wir y_{t+1} brauchen
        if oos_end_eff < oos_start:
            break

        _progress(
            f"[Stage A][Block {block_id}] train_end={train_end}, OOS={oos_start}-{oos_end_eff} | configs={len(survivors)}")
        preds_records: List[Dict[str, Any]] = []
        rmse_records: List[Dict[str, Any]] = []

        for i, hp in enumerate(survivors, start=1):
            _progress(f"  - Config {i}/{len(survivors)}")
            y_true_block, y_pred_block = [], []
            n_months = (oos_end_eff - oos_start + 1)

            # -------------------- Effektive FE/DR-Parameter --------------------
            L_eff = tuple(hp.get("lag_candidates", getattr(cfg, "lag_candidates", (1, 2, 3, 6, 12))))
            topk_lags_eff = int(hp.get("top_k_lags_per_feature", getattr(cfg, "top_k_lags_per_feature", 1)))
            use_rm3_eff = bool(hp.get("use_rm3", getattr(cfg, "use_rm3", True)))
            k1_topk_eff = int(hp.get("k1_topk", getattr(cfg, "k1_topk", 200)))
            screen_threshold_eff = hp.get("screen_threshold", getattr(cfg, "screen_threshold", None))
            redundancy_method_eff = str(hp.get("redundancy_method", getattr(cfg, "redundancy_method", "greedy")))
            redundancy_param_eff = float(hp.get("redundancy_param", getattr(cfg, "redundancy_param", 0.90)))
            dr_method_eff = str(hp.get("dr_method", getattr(cfg, "dr_method", "none")))
            pca_var_target_eff = float(hp.get("pca_var_target", getattr(cfg, "pca_var_target", 0.95)))
            pca_kmax_eff = int(hp.get("pca_kmax", getattr(cfg, "pca_kmax", 25)))
            pls_components_eff = int(hp.get("pls_components", getattr(cfg, "pls_components", 2)))
            target_block_set_eff = hp.get("target_block_set")
            hp_corr = hp.get("corr_spec", cfg.corr_spec)

            # ---- Fit einmalig auf Trainingsfenster bis train_end ----
            I_t = train_end + 1  # I_t ist exklusiv (wie im Origin-Fit)

            # 1) Lag-Selektions-Map (train-only)
            lag_map, _, D, taus = select_lags_per_feature(
                X, y, I_t=I_t, L=L_eff, k=topk_lags_eff,
                corr_spec=hp_corr,
            )

            # 2) Engineer design (lags + optional RM3)
            X_eng = build_engineered_matrix(X, lag_map)
            if use_rm3_eff:
                X_eng = apply_rm3(X_eng)

            # 2b) Target-only anhängen (Parquet)
            X_aug = _augment_with_target_blocks(X_eng, target_block_set_eff)

            # Head-Trim VOR Screening/Redundanz
            def _lag_of(col):
                try:
                    if "__lag" not in str(col):
                        return 0
                    return int(str(col).split('__lag')[-1])
                except Exception:
                    return 0

            max_lag_used = max([_lag_of(c) for c in X_eng.columns] + [0])
            rm_extra = 2 if use_rm3_eff else 0
            head_needed = max_lag_used + rm_extra

            taus_base_A = np.arange(1, int(I_t), dtype=int)

            # KORRIGIERT: Head-Trim Off-by-One (Kritik 1)
            taus_scr_mask = (taus_base_A - head_needed >= 0)
            if np.sum(taus_scr_mask) == 0:
                taus_scr = taus_base_A[-1:].copy() if taus_base_A.size > 0 else np.array([], dtype=int)
            else:
                taus_scr = taus_base_A[taus_scr_mask]

            # 3) Prewhitened Screening (train-only)
            keep_cols, scores = screen_k1(
                X_eng=X_aug, y=y, I_t=I_t, corr_spec=hp_corr, D=D, taus=taus_scr,
                k1_topk=k1_topk_eff, threshold=screen_threshold_eff
            )
            X_sel = X_aug.loc[:, keep_cols]

            # 4) Redundanz (train-only)
            if redundancy_method_eff == "greedy":
                kept = redundancy_reduce_greedy(
                    X_sel, hp_corr, D, taus_scr, redundancy_param_eff, scores=scores
                )
                X_red = X_sel.loc[:, kept]
            else:
                X_red = X_sel

            # 5) Head-Trim nach maximalem Lag
            head_needed_final = max([_lag_of(c) for c in X_red.columns] + [0])

            # KORRIGIERT: Head-Trim Off-by-One (Kritik 1)
            taus_model_mask = (taus_base_A - head_needed_final >= 0)
            if np.sum(taus_model_mask) == 0:
                taus_model = taus_base_A[-1:].copy() if taus_base_A.size > 0 else np.array([], dtype=int)
            else:
                taus_model = taus_base_A[taus_model_mask]

            X_tr = X_red.iloc[taus_model, :].copy()
            y_tr = y.shift(-1).iloc[taus_model]

            # 6) DR fit (train-only) und Train-Projection
            dr_map = fit_dr(
                X_tr,
                method=dr_method_eff,
                pca_var_target=pca_var_target_eff,
                pca_kmax=pca_kmax_eff,
                pls_components=pls_components_eff
            )
            Xb_tr = transform_dr(dr_map, X_tr, y_tr, fit_pls=(dr_method_eff == "pls"))

            # 7) Modell einmal fitten (KORRIGIERT: Seed injizieren)
            hp_with_seed = dict(hp)
            hp_with_seed['seed'] = cfg.seed
            model = model_ctor(hp_with_seed)

            # --- Modifikation für Sample Weights ---
            weight_decay = hp.get("sample_weight_decay")
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
            # --- Ende Modifikation ---

            # ---- Vorhersagen für den gesamten Block mit demselben Fit ----
            # KORRIGIERT: OOS-Schleife (Kritik 2)
            # t läuft von oos_start-1 (z.B. 180) bis oos_end_eff-1 (z.B. 199)
            for t in range(oos_start - 1, oos_end_eff):
                x_eval = X_red.iloc[[t], :].copy()
                Xb_eval = transform_dr(dr_map, x_eval, fit_pls=False)
                y_hat = model.predict_one(Xb_eval)

                y_true = float(y.iloc[t + 1])
                y_true_block.append(y_true)
                y_pred_block.append(float(y_hat))

                done = len(y_true_block)
                # --- LOGGING WIEDER MONATLICH ---
                if (done % 5 == 0) or (done == n_months):  # Logge nicht jeden Monat
                    _progress(
                        f"    · Month {done}/{n_months} processed | running...RMSE={rmse(np.array(y_true_block), np.array(y_pred_block)):.4f}"
                    )

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

        # CSV-Exports pro Block
        preds_df = pd.DataFrame(preds_records)
        rmse_df = pd.DataFrame(rmse_records)
        preds_path = rs.out_stageA / f"block{block_id}" / "preds.csv"
        rmse_path = rs.out_stageA / f"block{block_id}" / "rmse.csv"
        append_csv(preds_path, preds_df)
        append_csv(rmse_path, rmse_df)

        # Konfigurationen protokollieren
        configs_records = [{"block": f"block{block_id}", "model": model_name,
                            "config_id": i, "config_json": json.dumps(hp)}
                           for i, hp in enumerate(survivors, start=1)]
        configs_df = pd.DataFrame(configs_records)
        configs_path = rs.out_stageA / f"block{block_id}" / "configs.csv"
        append_csv(configs_path, configs_df)

        # Halving mit Untergrenze
        rmse_df_sorted = rmse_df.sort_values("rmse", ascending=True)
        k_keep = max(min_survivors_per_block, int(np.ceil(len(survivors) * 0.5)))
        k_keep = min(k_keep, len(survivors))
        keep_ids = set(rmse_df_sorted["config_id"].head(k_keep).tolist())
        survivors = [hp for i, hp in enumerate(survivors, start=1) if i in keep_ids]
        _progress(f"[Stage A][Block {block_id}] kept {len(survivors)} configs (floor={min_survivors_per_block}).")

    # Finale Freeze: Top-K anhand aggregierter Block-Scores (Median)
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

    # Shortlist persistieren
    (rs.out_stageA / "shortlist.json").write_text(json.dumps(shortlist, indent=2))
    _progress(f"[Stage A] Shortlist saved with {len(shortlist)} configs.")
    return shortlist


# ------------------------ Stage B (KORRIGIERT & ERWEITERT) ------------------------

def run_stageB(
        model_name: str,
        model_ctor: Callable[[Dict[str, Any]], Any],
        shortlist: List[Dict[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cfg: GlobalConfig,
        max_months: Optional[int] = None
) -> None:
    """
    Stage B mit gefrorener Shortlist und Online-Auswahl.
    """
    rs = _mk_paths(model_name, cfg)
    T = len(y)

    # Policy-Parameter (KORRIGIERT: Lesen aus cfg)
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

    months_iter = [t for t in stageB_months(cfg, T) if (t + 1) < T]  # t bis T-2
    if max_months is not None:
        months_iter = months_iter[:max_months]

    # Rolling-Fehler (Liste von SEs) pro Config
    rolling_errors: Dict[int, List[float]] = {i: [] for i in range(len(shortlist))}

    # KORRIGIERTE _wrmse Funktion
    def _wrmse(i: int) -> float:
        """Exponentiell abgezinste RMSE über 'window' Fehler mit 'decay'."""
        # Annahme: rolling_errors[i] enthält bereits QUADRIERTE Fehler (SEs)
        errs = rolling_errors[i][-window:] if window > 0 else rolling_errors[i]
        if len(errs) == 0:
            return float("inf")
        # Jüngere Fehler höher gewichten (w[0] = ältestes, w[-1] = jüngstes)
        w = np.array([decay ** k for k in range(len(errs) - 1, -1, -1)], dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            return float("inf") if len(errs) > 0 else 0.0
        w /= w_sum

        # KORRIGIERTER FIX: errs sind bereits e_t^2
        mse_w = float(np.sum(w * np.array(errs, dtype=float)))
        return float(np.sqrt(mse_w))

    for t in months_iter:
        _progress(f"[Stage B] Month origin t={t} | evaluating {len(shortlist)} configs | active={active_idx + 1}")
        y_truth = float(y.iloc[t + 1])

        # Vorhersagen für alle Kandidaten
        yhat_by_cfg: List[Tuple[int, float, float, PredictionLog]] = []
        for i, hp in enumerate(shortlist):
            # ERWEITERTES LOGGING: _fit_predict_one_origin gibt jetzt Dict zurück
            result_dict = _fit_predict_one_origin(
                model_ctor=model_ctor, model_hp=hp,
                X=X, y=y, t_origin=t, cfg=cfg,
                corr_spec=hp.get("corr_spec", cfg.corr_spec)
            )
            y_hat = result_dict["y_pred"]
            se = (y_truth - y_hat) ** 2
            yhat_by_cfg.append((i, y_hat, se, result_dict))  # Speichere ganzes Dict

        # Rolling-Fehler aktualisieren (nur SEs speichern)
        for i, _, se, _ in yhat_by_cfg:
            rolling_errors[i].append(se)

        # ---- Auswahl nach decayed metric ----
        wrmse_win = [_wrmse(i) for i in range(len(shortlist))]

        # KORRIGIERT: Nur 'decayed_best' (argmin der decayed RMSE)
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

        # Preds-CSV: alle Kandidaten (ERWEITERTES LOGGING)
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
                # --- NEUE LOG-SPALTEN ---
                "n_features_sis": result_dict["n_features_sis"],
                "n_features_redundant": result_dict["n_features_redundant"],
                "n_dr_components": result_dict["n_dr_components"],
                "ifo_dispersion_t": result_dict["ifo_dispersion_t"],
                "chronos_sigma_t": result_dict["chronos_sigma_t"]
            })
        append_csv(monthly_preds_path, pd.DataFrame(rows))

        # Scores-CSV
        rows2 = []
        for i in range(len(shortlist)):  # Iteriere über Indizes
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
        # RMSE pro Config
        rmse_by_cfg = (dfp.groupby("config_id")["se"].mean() ** 0.5).reset_index()
        rmse_by_cfg.rename(columns={"se": "rmse_overall"}, inplace=True)
        # RMSE des aktiven Pfads
        active = dfp[dfp["is_active"] == True]
        rmse_active = float(((active["y_true"] - active["y_pred"]) ** 2).mean() ** 0.5) if len(active) else np.nan
        summary = rmse_by_cfg.copy()
        summary["model"] = model_name
        summary_path = monthly_dir.parent / "summary" / "summary.csv"  # In 'summary' Ordner
        summary_path.parent.mkdir(exist_ok=True)
        summary.to_csv(summary_path, index=False)
        with open(monthly_dir.parent / "summary" / "summary_active.txt", "w") as f:
            f.write(f"RMSE_active_overall,{rmse_active:.6f}\n")
        _progress(f"[Stage B] summary.csv & summary_active.txt geschrieben.")
    except Exception as e:
        _progress(f"[Stage B] Summary-Schreiben übersprungen: {e}")