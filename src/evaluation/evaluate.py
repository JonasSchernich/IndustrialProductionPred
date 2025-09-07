# src/scripts/evaluate.py
from __future__ import annotations
import argparse, json, pathlib
import pandas as pd
from src.evaluation.splitters import ExpandingWindowSplit
from src.evaluation.backtest import run_backtest
from src.tuning.search_spaces import build_estimator_fn


def main():
    ap = argparse.ArgumentParser(description="Evaluate a tuned configuration with OOS backtest.")
    ap.add_argument("--features", type=str, default="data/processed/features.csv")
    ap.add_argument("--target", type=str, default="data/processed/target.csv")
    ap.add_argument("--config", type=str, required=True, help="Path to *_best_config.json from tuning.")
    ap.add_argument("--initial-window", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="reports")
    args = ap.parse_args()

    # Daten
    X = pd.read_csv(args.features, index_col=0, parse_dates=True)
    y = pd.read_csv(args.target, index_col=0, parse_dates=True).iloc[:, 0]
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    # Config laden
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    model_params = cfg["model_params"]
    feature_params = cfg["feature_params"]
    metric = cfg.get("metric", "mae")
    device = cfg.get("device", "cpu")

    # Estimator-Factory
    est_factory = build_estimator_fn(model_params, device=device)

    splitter = ExpandingWindowSplit(
        initial_window=args.initial_window,
        step=args.step,
        horizon=args.horizon,
    )

    preds_df, metrics_df = run_backtest(
        X=X, y=y, splitter=splitter,
        feature_params=feature_params,
        estimator_fn=est_factory,
        metric=metric,
    )

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Speichern
    preds_path = outdir / f"backtest_predictions_{model_name}.csv"
    metrics_path = outdir / f"overall_metrics_{model_name}.csv"

    preds_df.to_csv(preds_path)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[OK] Predictions → {preds_path}")
    print(f"[OK] Metrics     → {metrics_path}")


if __name__ == "__main__":
    main()
