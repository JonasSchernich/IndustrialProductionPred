# src/scripts/export_results.py
from __future__ import annotations
import argparse
import pathlib
import json
import pandas as pd
from typing import List
from src.evaluation.metrics import mae, rmse, smape, dm_test


def _collect_prediction_files(reports_dir: pathlib.Path) -> List[pathlib.Path]:
    return sorted(reports_dir.glob("backtest_predictions_*.csv"))


def summarize_predictions(pred_path: pathlib.Path) -> pd.Series:
    df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    s = pd.Series({
        "file": pred_path.name,
        "n_obs": len(df),
        "mae": mae(df["y_true"], df["y_pred"]),
        "rmse": rmse(df["y_true"], df["y_pred"]),
        "smape": smape(df["y_true"], df["y_pred"]),
    })
    return s


def main():
    ap = argparse.ArgumentParser(description="Aggregate tuning/backtest results into summary tables.")
    ap.add_argument("--reports", type=str, default="reports", help="Directory with backtest_predictions_*.csv")
    ap.add_argument("--tuned", type=str, default="models/tuned_params", help="Directory with *_trials.csv and *_best_config.json")
    ap.add_argument("--baseline", type=str, default=None, help="Model name in reports to use as DM baseline (e.g., 'elasticnet').")
    ap.add_argument("--out", type=str, default="reports/tables")
    args = ap.parse_args()

    reports_dir = pathlib.Path(args.reports)
    tuned_dir = pathlib.Path(args.tuned)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Backtest summaries
    pred_files = _collect_prediction_files(reports_dir)
    if not pred_files:
        print("[WARN] No prediction files found.")
        return

    rows = [summarize_predictions(p) for p in pred_files]
    summary = pd.DataFrame(rows).sort_values(by="mae")
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(f"[OK] summary_metrics.csv → {out_dir / 'summary_metrics.csv'}")

    # 2) Optional DM-Test gegen Baseline
    if args.baseline:
        base_file = reports_dir / f"backtest_predictions_{args.baseline}.csv"
        if not base_file.exists():
            print(f"[WARN] Baseline file not found: {base_file.name} — skipping DM tests.")
        else:
            base = pd.read_csv(base_file, index_col=0, parse_dates=True)
            dmt = []
            for p in pred_files:
                name = p.stem.replace("backtest_predictions_", "")
                if name == args.baseline:
                    continue
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                # Align auf gemeinsame Timestamps
                idx = base.index.intersection(df.index)
                if len(idx) < 3:
                    continue
                y = base.loc[idx, "y_true"]
                y1 = df.loc[idx, "y_pred"]
                y0 = base.loc[idx, "y_pred"]
                stat, pval = dm_test(y_true=y, y_pred_1=y1, y_pred_2=y0, h=1, loss="mse")
                dmt.append({"model": name, "baseline": args.baseline, "dm_stat": stat, "p_value": pval, "n": len(idx)})
            if dmt:
                dm_df = pd.DataFrame(dmt).sort_values("p_value")
                dm_df.to_csv(out_dir / f"dm_tests_vs_{args.baseline}.csv", index=False)
                print(f"[OK] DM tests → {out_dir / f'dm_tests_vs_{args.baseline}.csv'}")

    # 3) Tuning trials (optional, wenn vorhanden)
    trial_files = sorted(tuned_dir.glob("*_trials.csv"))
    if trial_files:
        trials = []
        for f in trial_files:
            df = pd.read_csv(f)
            df.insert(0, "model", f.stem.replace("_trials", ""))
            trials.append(df)
        all_trials = pd.concat(trials, ignore_index=True)
        all_trials.to_csv(out_dir / "all_tuning_trials.csv", index=False)
        print(f"[OK] all_tuning_trials.csv → {out_dir / 'all_tuning_trials.csv'}")


if __name__ == "__main__":
    main()
