# src/tuning.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .config import GlobalConfig, outputs_for_model
from .io_timesplits import stageA_blocks, stageB_months, append_csv
from .features import (
    select_lags_per_feature, build_engineered_matrix, apply_rm3,
    screen_k1, redundancy_reduce_greedy, fit_dr, transform_dr,
    tsfresh_block, chronos_block
)
from .evaluation import rmse


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


# ------------------------ Core: ein Origin schätzen ------------------------

def _fit_predict_one_origin(
    model_ctor: Callable[[Dict[str, Any]], Any],
    model_hp: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    t_origin: int,
    cfg: GlobalConfig,
    corr_spec,
    seasonal: str
) -> float:
    """
    Train-only Design bis inkl. Origin t (0-basiert), fit Modell, prognostiziere y_{t+1}.
    Wichtig: I_t = t_origin + 1 (Anzahl verfügbarer Zeilen bis inkl. t_origin).
    FE- und DR-Parameter werden 'effektiv' aus model_hp (falls vorhanden) oder cfg gelesen.
    """
    I_t = t_origin + 1  # 1-basiertes Zählen für die FE-Helfer

    # -------------------- Effektive FE/DR-Parameter pro Konfiguration --------------------
    # Lags
    L_eff = tuple(model_hp.get("lag_candidates", getattr(cfg, "lag_candidates", (1, 2, 3, 6, 12))))
    topk_lags_eff = int(model_hp.get("top_k_lags_per_feature", getattr(cfg, "top_k_lags_per_feature", 1)))

    # Short smoother
    use_rm3_eff = bool(model_hp.get("use_rm3", getattr(cfg, "use_rm3", False)))

    # Screening / Redundanz
    k1_topk_eff = int(model_hp.get("k1_topk", getattr(cfg, "k1_topk", 200)))
    screen_threshold_eff = model_hp.get("screen_threshold", getattr(cfg, "screen_threshold", None))
    redundancy_method_eff = str(model_hp.get("redundancy_method", getattr(cfg, "redundancy_method", "greedy")))
    redundancy_param_eff = float(model_hp.get("redundancy_param", getattr(cfg, "redundancy_param", 0.90)))

    # DR
    dr_method_eff = str(model_hp.get("dr_method", getattr(cfg, "dr_method", "none")))
    pca_var_target_eff = float(model_hp.get("pca_var_target", getattr(cfg, "pca_var_target", 0.95)))
    pca_kmax_eff = int(model_hp.get("pca_kmax", getattr(cfg, "pca_kmax", 50)))
    pls_components_eff = int(model_hp.get("pls_components", getattr(cfg, "pls_components", 4)))

    # -------------------- 1) Lag-Selektion --------------------
    lag_map, _, D, taus = select_lags_per_feature(
        X=X, y=y, I_t=I_t, L=L_eff, k=topk_lags_eff,
        corr_spec=corr_spec, seasonal_policy=seasonal
    )

    # -------------------- 2) Feature Engineering --------------------
    X_eng = build_engineered_matrix(X, lag_map)
    if use_rm3_eff:
        X_eng = apply_rm3(X_eng)

    # -------------------- 3) Screening K1 (prewhitened) --------------------
    keep_cols, scores = screen_k1(
        X_eng=X_eng, y=y, I_t=I_t, corr_spec=corr_spec, D=D, taus=taus,
        k1_topk=k1_topk_eff, threshold=screen_threshold_eff
    )
    X_sel = X_eng.loc[:, keep_cols]

    # -------------------- 4) Redundanzreduktion --------------------
    if redundancy_method_eff == "greedy":
        kept = redundancy_reduce_greedy(
            X_sel, corr_spec, D, taus, redundancy_param_eff, scores=scores
        )
        X_red = X_sel.loc[:, kept]
    else:
        # Platzhalter: Cluster/mRMR-Implementierung könnte hier angeschlossen werden
        X_red = X_sel

    # -------------------- 5) Head-Trim / Train-Design --------------------
    def _lag_of(col):
        try:
            return int(str(col).split("__lag")[-1])
        except Exception:
            return 0

    max_lag_used = max([_lag_of(c) for c in X_red.columns] + [0])
    rm_extra = 2 if use_rm3_eff else 0  # RM3 (Fenster 3) braucht 2 zusätzliche Perioden
    head_needed = max_lag_used + rm_extra

    taus_model = taus[taus - head_needed >= 0] if head_needed > 0 else taus
    if len(taus_model) == 0:
        taus_model = taus[-1:].copy()

    X_tr = X_red.iloc[taus_model, :].copy()
    y_tr = y.shift(-1).iloc[taus_model]

    # Eval-Design: Zeile t_origin
    x_eval = X_red.iloc[[t_origin], :].copy()

    # -------------------- 6) (Optional) Target-only Blöcke (Eval only; bewusst 'deaktiv' im Train) --------------------
    if getattr(cfg, "use_target_blocks", False):
        z_ts = tsfresh_block(y, I_t=I_t, W=12)   # inline @ t
        z_ch = chronos_block(y, I_t=I_t, W=12)
        x_eval = pd.concat([x_eval.reset_index(drop=True),
                            z_ts.reset_index(drop=True),
                            z_ch.reset_index(drop=True)], axis=1)

    # -------------------- 7) DR fit (train-only) und anwenden --------------------
    dr_map = fit_dr(
        X_tr,
        method=dr_method_eff,
        pca_var_target=pca_var_target_eff,
        pca_kmax=pca_kmax_eff,
        pls_components=pls_components_eff
    )
    if dr_method_eff == "pls":
        Xb_tr = transform_dr(dr_map, X_tr, y=y_tr, fit_pls=True)
        Xb_ev = transform_dr(dr_map, x_eval, y=None, fit_pls=False)
    else:
        Xb_tr = transform_dr(dr_map, X_tr)
        Xb_ev = transform_dr(dr_map, x_eval)

    # -------------------- 8) Modell fitten & Prognose --------------------
    model = model_ctor(model_hp)
    model.fit(np.asarray(Xb_tr), np.asarray(y_tr).ravel())
    y_hat = float(model.predict_one(np.asarray(Xb_ev)))
    return y_hat


# ------------------------ Stage A ------------------------

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
    Stage A mit drei Blöcken. Pro Block: je Konfiguration einmalig auf dem Trainingsfenster bis
    train_end fitten und mit demselben Fit über den gesamten Block OOS vorhersagen.
    CSV-Exports (preds, rmse, configs), dann Halving mit Untergrenze → nächste Stufe.
    Am Ende: gefrorene Shortlist (Top-K nach Median-RMSE über die drei Blöcke).
    """
    agg_scores: Dict[str, List[float]] = {}
    hp_by_key: Dict[str, Any] = {}

    rs = _mk_paths(model_name, cfg)
    T = len(y)
    survivors: List[Dict[str, Any]] = list(model_grid)

    for (train_end, oos_start, oos_end, block_id) in stageA_blocks(cfg, T):
        # Sicherheit: OOS-Ende muss y_{t+1} verfügbar lassen
        oos_end_eff = min(oos_end, T - 2)
        if oos_end_eff < oos_start:
            break

        _progress(f"[Stage A][Block {block_id}] train_end={train_end}, OOS={oos_start}-{oos_end_eff} | configs={len(survivors)}")
        preds_records: List[Dict[str, Any]] = []
        rmse_records:  List[Dict[str, Any]] = []

        # Bewertung je Kandidat (ein Fit pro Block & Config)
        for i, hp in enumerate(survivors, start=1):
            _progress(f"  - Config {i}/{len(survivors)}: {hp}")
            y_true_block, y_pred_block = [], []
            n_months = (oos_end_eff - oos_start + 1)

            # -------------------- Effektive FE/DR-Parameter (nur lesen; Fit bleibt unten zentral) --------------------
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

            # ---- Fit einmalig auf Trainingsfenster bis train_end ----
            I_t = train_end  # Origin = letztes Trainingsmonth im Block-Setup

            # 1) Lag-Selektions-Map (train-only)
            lag_map, _, D, taus = select_lags_per_feature(
                X, y, I_t=I_t, L=L_eff, k=topk_lags_eff,
                corr_spec=cfg.corr_spec, seasonal_policy=cfg.nuisance_seasonal
            )

            # 2) Engineer design (lags + optional RM3)
            X_eng = build_engineered_matrix(X, lag_map)
            if use_rm3_eff:
                X_eng = apply_rm3(X_eng)

            # 3) Prewhitened Screening (train-only)
            keep_cols, scores = screen_k1(
                X_eng, y, I_t, cfg.corr_spec, D, taus,
                k1_topk=k1_topk_eff, threshold=screen_threshold_eff
            )
            X_sel = X_eng.loc[:, keep_cols]

            # 4) Redundanz (train-only), Greedy nach absteigendem Score
            if redundancy_method_eff == "greedy":
                kept = redundancy_reduce_greedy(
                    X_sel, cfg.corr_spec, D, taus, redundancy_param_eff, scores=scores
                )
                X_red = X_sel.loc[:, kept]
            else:
                X_red = X_sel  # (Cluster/mRMR könnte hier später ergänzt werden)

            # 5) Head-Trim nach maximalem Lag
            def _lag_of(col):
                try:
                    return int(str(col).split("__lag")[-1])
                except Exception:
                    return 0
            head_needed = max([_lag_of(c) for c in X_red.columns] + [0])
            taus_model = taus[taus - head_needed >= 0] if head_needed > 0 else taus
            if len(taus_model) == 0:
                taus_model = taus[-1:].copy()

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

            # 7) Modell einmal fitten (mit internem Dev-Tail für ES im Backend)
            model = model_ctor(hp)
            model.fit(Xb_tr, y_tr.to_numpy(dtype=float))

            # ---- Vorhersagen für den gesamten Block mit demselben Fit ----
            for t in range(oos_start - 1, oos_end_eff + 0):  # 0-based t
                x_eval = X_red.iloc[[t], :].copy()
                Xb_eval = transform_dr(dr_map, x_eval, fit_pls=False)
                y_hat = model.predict_one(Xb_eval)

                y_true = float(y.iloc[t + 1])
                y_true_block.append(y_true)
                y_pred_block.append(float(y_hat))

                done = len(y_true_block)
                _progress(
                    f"    · Month {done}/{n_months} processed | running...MSE={rmse(np.array(y_true_block), np.array(y_pred_block)):.4f}"
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
            # Aggregation für Shortlist-Auswahl:
            key = json.dumps(hp, sort_keys=True)
            agg_scores.setdefault(key, []).append(float(score))
            hp_by_key[key] = hp

        # CSV-Exports pro Block
        preds_df = pd.DataFrame(preds_records)
        rmse_df  = pd.DataFrame(rmse_records)
        preds_path = rs.out_stageA / f"block{block_id}" / "preds.csv"
        rmse_path  = rs.out_stageA / f"block{block_id}" / "rmse.csv"
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


# ------------------------ Stage B ------------------------

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
    Stage B mit gefrorener Shortlist. Pro Monat: alle Kandidaten evaluieren.
    Auswahl/Messung über exponentiell gewichtete Fenster-Fehler (decayed metric),
    Guardrails (Gain + Cooldown), CSV-Exports.
    Zusätzlich wird am Ende eine summary.csv mit Overall-RMSE (pro Config & aktivem Pfad) geschrieben.
    """
    rs = _mk_paths(model_name, cfg)
    T = len(y)

    # Policy-Parameter
    window = int(getattr(cfg, "policy_window", 12))                # Fensterlänge (Monate)
    decay = float(getattr(cfg, "policy_decay", 0.95))              # Zerfallsfaktor λ in (0,1)
    gain_min = float(getattr(cfg, "policy_gain_min", 0.03))        # Mindest-Gewinn ggü. incumbent
    cooldown = int(getattr(cfg, "policy_cooldown", 3))             # Cooldown (Monate)
    selection_mode = str(getattr(cfg, "selection_mode", "decayed_best")).lower()
    weight_floor = float(getattr(cfg, "weight_floor", 1e-6))       # für evtl. Ensemble-Gewichte

    active_idx = 0
    last_switch_t: Optional[int] = None

    monthly_dir = rs.out_stageB / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    monthly_scores_path = monthly_dir / "scores.csv"
    monthly_preds_path  = monthly_dir / "preds.csv"

    months_iter = [t for t in stageB_months(cfg, T) if (t + 1) < T]
    if max_months is not None:
        months_iter = months_iter[:max_months]

    # Rolling-Fehler (Liste von SEs) pro Config
    rolling_errors: Dict[int, List[float]] = {i: [] for i in range(len(shortlist))}

    def _wrmse(i: int) -> float:
        """Exponentiell gewichtete Fenster-RMSE für Config i (höheres Gewicht auf jüngere Monate)."""
        arr = rolling_errors[i]
        n = len(arr)
        if n == 0:
            return float("inf")
        # arr ist chronologisch: älteste ... neueste (wir append() und pop(0))
        # Gewichte: älteste λ^(n-1), ..., neueste λ^0 = 1.0
        w = np.array([decay ** (n - 1 - k) for k in range(n)], dtype=float)
        se = np.array(arr, dtype=float)
        wmse = float(np.average(se, weights=w))
        return float(np.sqrt(wmse))

    for t in months_iter:
        _progress(f"[Stage B] Month origin t={t} | evaluating {len(shortlist)} configs | active={active_idx+1}")
        y_truth = float(y.iloc[t + 1])

        # Vorhersagen für alle Kandidaten (jeweils inkl. FE/DR aus hp)
        yhat_by_cfg = []
        for i, hp in enumerate(shortlist):
            y_hat = _fit_predict_one_origin(
                model_ctor=model_ctor, model_hp=hp,
                X=X, y=y, t_origin=t, cfg=cfg,
                corr_spec=cfg.corr_spec, seasonal=cfg.nuisance_seasonal
            )
            se = (y_truth - y_hat) ** 2
            yhat_by_cfg.append((i, y_hat, se))

        # Rolling-Fehler aktualisieren (Fenstergröße begrenzen)
        for i, _, se in yhat_by_cfg:
            rolling_errors[i].append(se)
            if len(rolling_errors[i]) > window:
                # ältesten Eintrag entfernen
                rolling_errors[i].pop(0)

        # ---- Auswahl nach decayed metric ----
        wrmse_win = [ _wrmse(i) for i in range(len(shortlist)) ]

        if selection_mode == "decayed_ensemble":
            # Optionaler Ensemble-Pfad (gewichtetes Mittel mit inversen decayed errors)
            eps = 1e-12
            inv = np.array([1.0 / (eps + (wrmse if np.isfinite(wrmse) else 1e6)) for wrmse in wrmse_win], dtype=float)
            inv = np.maximum(inv, weight_floor)
            w = inv / np.sum(inv)
            # (Wir behalten weiterhin 'active_idx' für Logging/Guardrails; Use-Case: Ensemble nur zur Stabilisierung)
            new_idx = int(np.argmax(w))  # „repräsentativer“ Index (größtes Gewicht)
            new_rmse = float(np.sum(w * np.array([_wrmse(i) for i in range(len(shortlist))], dtype=float)))
        else:
            # Standard: decayed_best – nimm die beste einzelne Konfiguration
            new_idx = int(np.argmin(wrmse_win))
            new_rmse = wrmse_win[new_idx]

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

        # Preds-CSV: alle Kandidaten
        rows = []
        for i, y_hat, _ in yhat_by_cfg:
            rows.append({
                "t": t, "date_t_plus_1": y.index[t+1].strftime("%Y-%m-%d"),
                "y_true": y_truth, "y_pred": y_hat,
                "model": model_name, "config_id": i+1,
                "is_active": (i == active_idx),
                "wrmse_window": wrmse_win[i],
                "window_len": len(rolling_errors[i]),
                "selection_mode": selection_mode
            })
        append_csv(monthly_preds_path, pd.DataFrame(rows))

        # Scores-CSV: decayed RMSE im Policy-Fenster + Guardrail-Flags
        rows2 = []
        for i in range(len(shortlist)):
            rows2.append({
                "t": t, "model": model_name, "config_id": i+1,
                "wrmse_window": wrmse_win[i], "window_len": len(rolling_errors[i]),
                "active_idx": active_idx + 1,
                "candidate_best_idx": new_idx + 1,
                "gain_vs_incumbent": rel_gain if i == new_idx else 0.0,
                "cooldown": cooldown, "cooldown_ok": (last_switch_t is None) or ((t - (last_switch_t or 0)) >= cooldown),
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
        # RMSE pro Config über die ganze Stage-B-OOS:
        rmse_by_cfg = (dfp.groupby("config_id")["se"].mean() ** 0.5).reset_index()
        rmse_by_cfg.rename(columns={"se": "rmse_overall"}, inplace=True)
        # RMSE des aktiven Pfads:
        active = dfp[dfp["is_active"] == True]
        rmse_active = float(((active["y_true"] - active["y_pred"]) ** 2).mean() ** 0.5) if len(active) else np.nan
        summary = rmse_by_cfg.copy()
        summary["model"] = model_name
        summary_path = monthly_dir / "summary.csv"
        # Schreibe Summary und hänge eine Zeile für den aktiven Pfad an
        summary.to_csv(summary_path, index=False)
        with open(monthly_dir / "summary_active.txt", "w") as f:
            f.write(f"RMSE_active_overall,{rmse_active:.6f}\n")
        _progress(f"[Stage B] summary.csv & summary_active.txt geschrieben.")
    except Exception as e:
        _progress(f"[Stage B] Summary-Schreiben übersprungen: {e}")
