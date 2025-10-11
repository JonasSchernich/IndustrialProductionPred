from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

from .rolling.online import online_rolling_forecast
from .types import FeatureSelectCfg, FeEngCfg, TrainEvalCfg, AshaCfg, BOCfg, EnsembleCfg
from .utils.reporting import append_predictions_rows


def run_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_grid: Dict,
    fs_cfg: FeatureSelectCfg,
    fe_cfg: FeEngCfg,
    initial_window: int,
    step: int = 1,
    horizon: int = 1,
    train_eval_cfg: Optional[TrainEvalCfg] = None,
    asha_cfg: Optional[AshaCfg] = None,
    bo_cfg: Optional[BOCfg] = None,
    per_feature_lag_refresh_k: Optional[int] = None,
    report_csv_path: Optional[str] = None,
    tuning_csv_path: Optional[str] = None,
    predictions_csv_path: Optional[str] = None,
    metric_fn: Optional[Callable] = None,
    progress: bool = False,
    progress_fn: Optional[Callable] = None,
    # neu
    min_rel_improvement: float = 0.0,
    inner_cv_cfg: Optional[Dict] = None,
    policy_cfg: Optional[Dict] = None,
):
    if metric_fn is None:
        def metric_fn(a, b):
            a = np.asarray(a, float)
            b = np.asarray(b, float)
            return float(np.sqrt(np.mean((a - b) ** 2)))

    preds, truths, cfgdf = online_rolling_forecast(
        X=X, y=y,
        initial_window=initial_window, step=step, horizon=horizon,
        fs_cfg=fs_cfg, fe_cfg=fe_cfg,
        model_name=model_name, model_grid=model_grid, metric_fn=metric_fn,
        progress=progress, progress_fn=progress_fn,
        per_feature_lag_refresh_k=per_feature_lag_refresh_k,
        train_eval_cfg=train_eval_cfg,
        asha_cfg=asha_cfg, bo_cfg=bo_cfg,
        report_csv_path=report_csv_path,
        tuning_csv_path=tuning_csv_path,
        predictions_csv_path=predictions_csv_path,
        # weiterreichen
        min_rel_improvement=min_rel_improvement,
        inner_cv_cfg=inner_cv_cfg,
        policy_cfg=policy_cfg,
    )
    return preds, truths, cfgdf


def weighted_mean_ensemble(pred_map: Dict[str, pd.Series], truth: pd.Series) -> pd.Series:
    models = list(pred_map.keys())
    rmses = []
    for m in models:
        p = pred_map[m].reindex(truth.index).astype(float)
        rmse = float(np.sqrt(np.mean((p.values - truth.values) ** 2)))
        rmses.append(max(rmse, 1e-12))
    inv = np.reciprocal(np.asarray(rmses))
    w = inv / inv.sum()
    P = np.zeros_like(truth.values, dtype=float)
    for wi, m in zip(w, models):
        P += wi * pred_map[m].reindex(truth.index).values.astype(float)
    return pd.Series(P, index=truth.index)


def stacking_ensemble_rolling(
    pred_map: Dict[str, pd.Series],
    truth: pd.Series,
    meta_model: str = "elasticnet",           # "elasticnet" | "lgbm"
    meta_params: Optional[Dict] = None,
    min_train: int = 24,
    predictions_csv_path: Optional[str] = None,
    tag: str = "stacking"
) -> pd.Series:
    """Strikt OOS: Für jeden Zeitpunkt t wird der Meta-Lerner nur auf {<=t-1} trainiert."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = sorted(pred_map.keys())
    T = truth.index
    X_meta = pd.DataFrame({m: pred_map[m].reindex(T).astype(float) for m in models}, index=T)
    y = truth.astype(float)

    if meta_model.lower() == "elasticnet":
        from sklearn.linear_model import ElasticNet
        p = dict(alpha=0.01, l1_ratio=0.1, max_iter=5000, random_state=42)
        if meta_params:
            p.update({k: v for k, v in meta_params.items() if v is not None})
        meta = Pipeline([("scaler", StandardScaler()), ("en", ElasticNet(**p))])
        fit_fn = lambda X, y: meta.fit(X, y)
        pred_fn = lambda X: meta.predict(X)
    elif meta_model.lower() in ("lgbm", "lightgbm"):
        import lightgbm as lgb
        p = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                 colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        if meta_params:
            p.update({k: v for k, v in meta_params.items() if v is not None})
        meta = lgb.LGBMRegressor(**p)
        fit_fn = lambda X, y: meta.fit(X, y)
        pred_fn = lambda X: meta.predict(X)
    else:
        raise ValueError("meta_model must be 'elasticnet' or 'lgbm'")

    y_meta_hat = []
    rows_buf = []

    for i, t in enumerate(T):
        if i < min_train:
            y_meta_hat.append(np.nan)
            continue
        X_tr = X_meta.iloc[:i, :].values
        y_tr = y.iloc[:i].values
        X_te = X_meta.iloc[i:i+1, :].values
        try:
            fit_fn(X_tr, y_tr)
            yh = float(pred_fn(X_te)[0])
        except Exception:
            yh = np.nan
        y_meta_hat.append(yh)

        if predictions_csv_path is not None:
            rows_buf.append({"time": t, "model": tag, "y_true": float(y.loc[t]), "y_hat": yh})
            if len(rows_buf) >= 500:
                append_predictions_rows(predictions_csv_path, rows_buf)
                rows_buf.clear()

    if predictions_csv_path is not None and rows_buf:
        append_predictions_rows(predictions_csv_path, rows_buf)
        rows_buf.clear()

    return pd.Series(y_meta_hat, index=T)


def run_models_and_stack(
    X: pd.DataFrame,
    y: pd.Series,
    model_specs: List[Tuple[str, Dict]],  # [(model_name, model_grid), ...]
    fs_cfg: FeatureSelectCfg,
    fe_cfg: FeEngCfg,
    initial_window: int,
    step: int = 1,
    horizon: int = 1,
    train_eval_cfg: Optional[TrainEvalCfg] = None,
    asha_cfg: Optional[AshaCfg] = None,
    bo_cfg: Optional[BOCfg] = None,
    per_feature_lag_refresh_k: Optional[int] = None,
    report_csv_path: Optional[str] = None,
    tuning_csv_path: Optional[str] = None,
    predictions_csv_path: Optional[str] = None,
    metric_fn: Optional[Callable] = None,
    progress: bool = False,
    progress_fn: Optional[Callable] = None,
    do_weighted_mean: bool = True,
    do_stacking: bool = True,
    stacking_meta_model: str = "elasticnet",
    stacking_meta_params: Optional[Dict] = None,
    stacking_min_train: int = 24,
    # neu -> durchreichen
    min_rel_improvement: float = 0.0,
    inner_cv_cfg: Optional[Dict] = None,
    policy_cfg: Optional[Dict] = None,
) -> Dict[str, pd.Series]:
    """Mehrere Basismodelle OOS ausführen, optional Ensembling."""
    pred_map: Dict[str, pd.Series] = {}
    truth_ref: Optional[pd.Series] = None

    for model_name, model_grid in model_specs:
        preds, truths, _ = online_rolling_forecast(
            X=X, y=y,
            initial_window=initial_window, step=step, horizon=horizon,
            fs_cfg=fs_cfg, fe_cfg=fe_cfg,
            model_name=model_name, model_grid=model_grid, metric_fn=metric_fn,
            progress=progress, progress_fn=progress_fn,
            per_feature_lag_refresh_k=per_feature_lag_refresh_k,
            train_eval_cfg=train_eval_cfg,
            asha_cfg=asha_cfg, bo_cfg=bo_cfg,
            report_csv_path=report_csv_path,
            tuning_csv_path=tuning_csv_path,
            predictions_csv_path=predictions_csv_path,
            min_rel_improvement=min_rel_improvement,
            inner_cv_cfg=inner_cv_cfg,
            policy_cfg=policy_cfg,
        )
        pred_map[model_name] = preds
        if truth_ref is None:
            truth_ref = truths
        else:
            common = truth_ref.index.intersection(truths.index)
            truth_ref = truth_ref.reindex(common)
            pred_map = {k: v.reindex(common) for k, v in pred_map.items()}

    out: Dict[str, pd.Series] = {}
    if truth_ref is None:
        return out

    out["truth"] = truth_ref

    if do_weighted_mean:
        wm = weighted_mean_ensemble(pred_map, truth_ref)
        out["weighted"] = wm
        if predictions_csv_path is not None:
            rows = [{"time": t, "model": "weighted", "y_true": float(truth_ref.loc[t]), "y_hat": float(wm.loc[t])}
                    for t in wm.index]
            append_predictions_rows(predictions_csv_path, rows)

    if do_stacking:
        st = stacking_ensemble_rolling(
            pred_map, truth_ref,
            meta_model=stacking_meta_model,
            meta_params=stacking_meta_params,
            min_train=stacking_min_train,
            predictions_csv_path=predictions_csv_path,
            tag="stacking"
        )
        out["stacking"] = st

    for m, s in pred_map.items():
        out[m] = s

    return out

