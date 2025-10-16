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
    """
    I_t = t_origin + 1  # 1-basiertes Zählen für die FE-Helfer

    # 1) Lag-Selektion (prewhitened, einheitliche Korrelation)
    lag_map, _, D, taus = select_lags_per_feature(
        X=X, y=y, I_t=I_t, L=list(cfg.lag_candidates), k=int(cfg.top_k_lags_per_feature),
        corr_spec=corr_spec, seasonal_policy=seasonal
    )

    # 2) Feature Engineering (Original + gewählte Lags) und optional RM3
    X_eng = build_engineered_matrix(X, lag_map)
    if getattr(cfg, "use_rm3", False):
        X_eng = apply_rm3(X_eng)

    # 3) Screening K1 (prewhitened) auf Trainingsfenster
    keep_cols, _scores = screen_k1(
        X_eng=X_eng, y=y, I_t=I_t, corr_spec=corr_spec, D=D, taus=taus,
        k1_topk=int(cfg.k1_topk), threshold=getattr(cfg, "screen_threshold", None)
    )
    X_sel = X_eng.loc[:, keep_cols]

    # 4) Redundanzreduktion (prewhitened)
    if getattr(cfg, "redundancy_method", "greedy") == "greedy":
        kept = redundancy_reduce_greedy(X_sel, corr_spec, D, taus, float(cfg.redundancy_param))
        X_red = X_sel.loc[:, kept]
    else:
        # Platzhalter für Cluster-Variante
        X_red = X_sel


    # ---------------------------------------------------------
    # maximal verwendeter Lag (über alle Features)
    try:
        max_lag_used = max([lag for lags in lag_map.values() for lag in lags if lag is not None], default=0)
    except ValueError:
        max_lag_used = 0
    # Zusatzfenster für RM3 (Fenster 3 → braucht 2 frühere Perioden)
    rm_extra = 2 if getattr(cfg, "use_rm3", False) else 0
    head_needed = max_lag_used + rm_extra

    if head_needed > 0:
        taus_model = taus[taus - head_needed >= 0]
    else:
        taus_model = taus

    if len(taus_model) == 0:
        # Fallback: wenigstens die letzte gültige Zeile
        taus_model = taus[-1:].copy()
    # Trainingsdesign exakt auf getrimmte τ
    X_red_tr = X_red.iloc[taus_model, :].copy()
    y_tr = y.shift(-1).iloc[taus_model]

    # Eval-Design: Zeile t_origin
    x_eval = X_red.iloc[[t_origin], :].copy()

    # =========================================================

    # 7) Target-only Blöcke nur wenn explizit aktiviert (hier nur Eval-Row angehängt)
    if getattr(cfg, "use_target_blocks", False):
        z_ts = tsfresh_block(y, I_t=I_t, W=12)   # inline @ t
        z_ch = chronos_block(y, I_t=I_t, W=12)
        x_eval = pd.concat([x_eval.reset_index(drop=True),
                            z_ts.reset_index(drop=True),
                            z_ch.reset_index(drop=True)], axis=1)
        # Hinweis: Für volle Wirkung optional auch trainseitig integrieren.

    # 8) DR fit (train-only) und anwenden
    dr_map = fit_dr(
        X_red_tr,
        method=getattr(cfg, "dr_method", "none"),
        pca_var_target=float(getattr(cfg, "pca_var_target", 0.95)),
        pca_kmax=int(getattr(cfg, "pca_kmax", 50)),
        pls_components=int(getattr(cfg, "pls_components", 4))
    )
    if getattr(cfg, "dr_method", "none") == "pls":
        X_tr_dr = transform_dr(dr_map, X_red_tr, y=y_tr, fit_pls=True)
        x_ev_dr = transform_dr(dr_map, x_eval, y=None, fit_pls=False)
    else:
        X_tr_dr = transform_dr(dr_map, X_red_tr)
        x_ev_dr = transform_dr(dr_map, x_eval)

    # 9) Modell fitten & Prognose
    model = model_ctor(model_hp)
    # X_tr, y_tr sind das train-only Design + Ziel (N x P)
    # --- Design-Sanity vor dem Fit (train-only) ---
    import numpy as np, pandas as pd

    import numpy as np, pandas as pd

    # --- Quick Corr Diagnostic: prüft, ob X_tr_dr überhaupt Signal zu y_tr trägt ---
    import numpy as np, pandas as pd



    model.fit(np.asarray(X_tr_dr), np.asarray(y_tr).ravel())
    y_hat = float(model.predict_one(np.asarray(x_ev_dr)))
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
    Stage A mit drei Blöcken. Pro Block: walk-forward Re-Schätzung je Konfiguration,
    CSV-Exports (preds, rmse, configs), dann Halving mit Untergrenze → nächste Stufe.
    Am Ende: gefrorene Shortlist (Top-K).
    """
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

        # Bewertung je Kandidat
        for i, hp in enumerate(survivors, start=1):
            _progress(f"  - Config {i}/{len(survivors)}: {hp}")
            y_true_block, y_pred_block = [], []
            n_months = (oos_end_eff - oos_start + 1)

            for t in range(oos_start - 1, oos_end_eff + 0):  # 0-based t; letztes t hat y_{t+1} im Bereich
                y_hat = _fit_predict_one_origin(
                    model_ctor=model_ctor, model_hp=hp,
                    X=X, y=y, t_origin=t, cfg=cfg,
                    corr_spec=cfg.corr_spec, seasonal=cfg.nuisance_seasonal
                )
                y_true = float(y.iloc[t + 1])
                y_true_block.append(y_true)
                y_pred_block.append(float(y_hat))

                done = len(y_true_block)
                _progress(f"    · Month {done}/{n_months} processed | running RMSE={rmse(np.array(y_true_block), np.array(y_pred_block)):.4f}")

                preds_records.append({
                    "block": f"block{block_id}", "t": t,
                    "date_t_plus_1": y.index[t+1].strftime("%Y-%m-%d"),
                    "y_true": y_true, "y_pred": y_hat,
                    "model": model_name, "config_id": i
                })

            score = rmse(np.array(y_true_block), np.array(y_pred_block))
            rmse_records.append({
                "block": f"block{block_id}", "model": model_name, "config_id": i,
                "rmse": score, "n_oos": len(y_true_block),
                "train_end": train_end, "oos_start": oos_start, "oos_end": oos_end_eff
            })

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

    # Finale Freeze: Top-K
    k_final = min(int(keep_top_k_final), len(survivors))
    shortlist = survivors[:k_final]

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
    Stage B mit gefrorener Shortlist. Pro Monat: alle Kandidaten evaluieren,
    Rolling-RMSE über Policy-Fenster, Guardrails (Gain + Cooldown), CSV-Exports.
    Zusätzlich wird am Ende eine summary.csv mit Overall-RMSE (pro Config & aktivem Pfad) geschrieben.
    """
    rs = _mk_paths(model_name, cfg)
    T = len(y)

    active_idx = 0
    last_switch_t: Optional[int] = None
    window = int(getattr(cfg, "policy_window", 12))
    gain_min = float(getattr(cfg, "policy_gain_min", 0.03))
    cooldown = int(getattr(cfg, "policy_cooldown", 3))

    monthly_dir = rs.out_stageB / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    monthly_scores_path = monthly_dir / "scores.csv"
    monthly_preds_path  = monthly_dir / "preds.csv"

    months_iter = [t for t in stageB_months(cfg, T) if (t + 1) < T]
    if max_months is not None:
        months_iter = months_iter[:max_months]

    # Rolling-Fehler pro Config
    rolling_errors: Dict[int, List[float]] = {i: [] for i in range(len(shortlist))}

    for t in months_iter:
        _progress(f"[Stage B] Month origin t={t} | evaluating {len(shortlist)} configs | active={active_idx+1}")
        y_truth = float(y.iloc[t + 1])

        # Vorhersagen für alle Kandidaten
        yhat_by_cfg = []
        for i, hp in enumerate(shortlist):
            y_hat = _fit_predict_one_origin(
                model_ctor=model_ctor, model_hp=hp,
                X=X, y=y, t_origin=t, cfg=cfg,
                corr_spec=cfg.corr_spec, seasonal=cfg.nuisance_seasonal
            )
            se = (y_truth - y_hat) ** 2
            yhat_by_cfg.append((i, y_hat, se))

        # Rolling-Fehler aktualisieren
        for i, _, se in yhat_by_cfg:
            rolling_errors[i].append(se)
            if len(rolling_errors[i]) > window:
                rolling_errors[i].pop(0)

        # Fenster-RMSE
        def _rm(i: int) -> float:
            arr = rolling_errors[i]
            return float(np.sqrt(np.mean(arr))) if len(arr) > 0 else np.inf

        inc_rmse = _rm(active_idx)
        rmse_win = [ _rm(i) for i in range(len(shortlist)) ]
        new_idx = int(np.argmin(rmse_win))
        new_rmse = rmse_win[new_idx]

        rel_gain = 0.0
        if np.isfinite(inc_rmse) and np.isfinite(new_rmse) and inc_rmse > 0:
            rel_gain = 1.0 - (new_rmse / inc_rmse)

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
                "is_active": (i == active_idx)
            })
        append_csv(monthly_preds_path, pd.DataFrame(rows))

        # Scores-CSV: RMSE im Policy-Fenster + Guardrail-Flags
        rows2 = []
        for i in range(len(shortlist)):
            rows2.append({
                "t": t, "model": model_name, "config_id": i+1,
                "rmse_window": rmse_win[i], "window_len": len(rolling_errors[i]),
                "active_idx": active_idx + 1,
                "candidate_best_idx": new_idx + 1,
                "gain_vs_incumbent": rel_gain if i == new_idx else 0.0,
                "cooldown": cooldown, "cooldown_ok": (last_switch_t is None) or ((t - (last_switch_t or 0)) >= cooldown),
                "switched": switched
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
