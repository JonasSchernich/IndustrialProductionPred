from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import optuna

from src.features.pipeline import make_feature_pipeline
from src.evaluation.metrics import get_metric
from src.evaluation.splitters import ExpandingWindowSplit
from .search_spaces import (
    load_space_yaml,
    suggest_feature_params, suggest_model_params,
    suggest_feature_params_from_yaml, suggest_model_params_from_yaml,
    build_estimator_fn,
)

def _dropna_align(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    # Nach Lags verwerfen wir Trainingszeilen, die IRGENDEIN NaN enthalten
    mask = ~X.isna().any(axis=1)
    if mask.sum() == 0:
        return X.iloc[:0], y.iloc[:0]
    return X.loc[mask], y.loc[mask]

def _as_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)

def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: ExpandingWindowSplit,
    *,
    model: str,
    n_trials: int = 50,
    metric: str = "mae",
    device: str = "cpu",
    random_seed: int = 42,
    pca_groups: Optional[Dict[str, list]] = None,
    groupwise_lags: Optional[Dict[str, list]] = None,
    study_name: Optional[str] = None,
    direction: str = "minimize",
    pruner: Optional[optuna.pruners.BasePruner] = None,
    space_yaml: Optional[Dict[str, Any]] = None,
) -> Tuple[optuna.study.Study, Dict[str, Any]]:

    scorer = get_metric(metric)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name)

    X = X.copy()
    y = _as_series(y).copy()

    folds = list(splitter.split(X))
    if len(folds) == 0:
        n = len(X)
        raise ValueError(
            f"ExpandingWindowSplit erzeugt 0 Folds (n={n}, initial_window={splitter.initial_window}, "
            f"horizon={splitter.horizon}, step={splitter.step}). "
            "Lösung: initial_window verkleinern oder längere Daten verwenden."
        )

    def objective(trial: optuna.trial.Trial) -> float:
        if space_yaml is None:
            fparams = suggest_feature_params(trial, pca_groups=pca_groups, groupwise_lags=groupwise_lags)
            mparams = suggest_model_params(trial, model=model, device=device)
        else:
            fparams = suggest_feature_params_from_yaml(trial, space_yaml, pca_groups=pca_groups, groupwise_lags=groupwise_lags)
            mparams = suggest_model_params_from_yaml(trial, space_yaml, model=model, device=device)

        est_factory = build_estimator_fn(mparams, device=device)
        scores: List[float] = []

        for tr_idx, te_idx in folds:
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            feat_pipe = make_feature_pipeline(**fparams)

            # Train
            Xtr_ft = pd.DataFrame(feat_pipe.fit_transform(Xtr, ytr), index=Xtr.index)
            Xtr_ft, ytr_al = _dropna_align(Xtr_ft, ytr)
            if Xtr_ft.shape[0] == 0 or Xtr_ft.shape[1] == 0:
                return 1e9

            # Fit & Predict
            est = est_factory()
            est.fit(Xtr_ft, ytr_al)

            # Test (LagMaker nutzt Buffer aus fit())
            Xte_ft = pd.DataFrame(feat_pipe.transform(Xte), index=Xte.index)
            mask = ~Xte_ft.isna().any(axis=1)
            if mask.sum() == 0:
                return 1e9
            yte_sub = yte.loc[mask]
            yhat = _as_series(est.predict(Xte_ft.loc[mask])); yhat.index = yte_sub.index
            scores.append(scorer(yte_sub, yhat))

        if not scores:
            return 1e9
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    class Frozen:
        def __init__(self, params: Dict[str, Any]): self.params = params
        def suggest_categorical(self, name, choices): return self.params[name]
        def suggest_float(self, name, low, high, *, log=False, step=None): return self.params[name]
        def suggest_int(self, name, low, high, *, step=1, log=False): return self.params[name]

    if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError("Keine erfolgreichen Trials. Prüfe Datenlängen, initial_window und YAML-Konfiguration.")

    frozen = Frozen(study.best_trial.params)
    if space_yaml is None:
        best_fparams = suggest_feature_params(frozen, pca_groups=pca_groups, groupwise_lags=groupwise_lags)
        best_mparams = suggest_model_params(frozen, model=model, device=device)
    else:
        best_fparams = suggest_feature_params_from_yaml(frozen, space_yaml, pca_groups=pca_groups, groupwise_lags=groupwise_lags)
        best_mparams = suggest_model_params_from_yaml(frozen, space_yaml, model=model, device=device)

    best_config = {
        "model": model,
        "model_params": best_mparams,
        "feature_params": best_fparams,
        "metric": metric,
        "value": study.best_value,
        "device": device,
        "n_trials": n_trials,
        "random_seed": random_seed,
    }
    return study, best_config

def save_study_results(study: optuna.study.Study, best_config: Dict[str, Any], out_dir: str, prefix: str) -> None:
    import pathlib
    p = pathlib.Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe(attrs=("number", "value", "params", "state")).to_csv(p / f"{prefix}_trials.csv", index=False)
    with open(p / f"{prefix}_best_config.json", "w", encoding="utf-8") as f:
        import json; json.dump(best_config, f, indent=2)
