# src/scripts/tuning.py
from __future__ import annotations
import argparse, pathlib
import pandas as pd
from src.tuning.tuner import tune_model, save_study_results
from src.tuning.search_spaces import load_space_yaml
from src.evaluation.splitters import ExpandingWindowSplit
from src.features.groups import infer_groups_from_dataframe

def main():
    ap = argparse.ArgumentParser(description="Run hyperparameter tuning with Expanding-Window CV.")
    ap.add_argument("--features", type=str, default="data/processed/features.csv")
    ap.add_argument("--target", type=str, default="data/processed/target.csv")
    ap.add_argument("--model", type=str, required=False, help="elasticnet|svr|rf|xgb|lgbm|pls (ignored if --space given)")
    ap.add_argument("--metric", type=str, default=None, help="override metric; else YAML or default=mae")
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda (CLI overrides YAML)")
    ap.add_argument("--initial-window", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="models/tuned_params")
    ap.add_argument("--space", type=str, default=None, help="Path to YAML search space (configs/search_spaces/*.yaml)")
    args = ap.parse_args()

    X = pd.read_csv(args.features, index_col=0, parse_dates=True)
    y = pd.read_csv(args.target, index_col=0, parse_dates=True).iloc[:, 0]
    common_index = X.index.intersection(y.index)
    X, y = X.loc[common_index], y.loc[common_index]
    if len(X) <= args.initial_window + args.horizon:
        raise SystemExit(
            f"Zu wenige gemeinsame Zeilen ({len(X)}). "
            f"Erforderlich: > initial_window({args.initial_window}) + horizon({args.horizon}). "
            "Setze --initial-window kleiner oder nutze längere Daten."
        )

    groups = infer_groups_from_dataframe(X)
    splitter = ExpandingWindowSplit(initial_window=args.initial_window, step=args.step, horizon=args.horizon)

    space = load_space_yaml(args.space) if args.space else None
    # Modell, Metric, Device bestimmen (CLI > YAML > Defaults)
    model_name = (space.get("model") if space else None) or args.model
    if model_name is None:
        raise SystemExit("Either --model or --space (with 'model: ...') must be provided.")
    metric = args.metric or (space.get("metric") if space else "mae")
    device = args.device or (space.get("device") if space else "cpu")

    study, best_cfg = tune_model(
        X=X, y=y, splitter=splitter,
        model=model_name,
        metric=metric,
        device=device,
        n_trials=args.trials,
        random_seed=args.seed,
        pca_groups=groups,
        groupwise_lags=groups,
        study_name=f"{model_name}_tuning",
        space_yaml=space,         # << YAML aktiviert; None = programmatic
    )

    out_dir = pathlib.Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    save_study_results(study, best_cfg, str(out_dir), prefix=model_name)
    print(f"[OK] Best config → {out_dir / f'{model_name}_best_config.json'}")
    print(f"[OK] Best {metric}: {best_cfg['value']:.6f}")

if __name__ == "__main__":
    main()
