# src/evaluation/backtest.py
from __future__ import annotations
from typing import Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.features.pipeline import make_feature_pipeline
from src.evaluation.metrics import get_metric
from src.evaluation.splitters import ExpandingWindowSplit


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: ExpandingWindowSplit,
    *,
    feature_params: Dict[str, Any],
    estimator_fn: Callable[[], Any],
    metric: str = "mae",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Führt einen Out-of-Sample Expanding-Window Backtest durch.
    Returns:
      preds_df: rows = Test-Zeitpunkte, cols = ['y_true','y_pred','train_end','fold']
      metrics_df: Metriken aggregiert über alle Folds
    """
    scorer = get_metric(metric)

    preds = []
    fold_no = 0
    for tr_idx, te_idx in splitter.split(X):
        fold_no += 1
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        feat_pipe = make_feature_pipeline(**feature_params)
        Xtr_ft = feat_pipe.fit_transform(Xtr, ytr)
        Xtr_ft = pd.DataFrame(Xtr_ft, index=Xtr.index)

        # Drop NaNs im Train
        mask_tr = ~np.any(pd.isna(Xtr_ft.values), axis=1)
        Xtr_ft = Xtr_ft.loc[mask_tr]
        ytr_al = ytr.loc[mask_tr]

        if Xtr_ft.shape[0] == 0 or Xtr_ft.shape[1] == 0:
            # überspringen (sollte nicht passieren, wenn Config valide ist)
            continue

        est = estimator_fn()
        est.fit(Xtr_ft, ytr_al)

        # Test-Features
        Xte_ft = feat_pipe.transform(Xte)
        Xte_ft = pd.DataFrame(Xte_ft, index=Xte.index)
        mask_te = ~np.any(pd.isna(Xte_ft.values), axis=1)
        if mask_te.sum() == 0:
            continue

        yte_sub = yte.loc[mask_te]
        yhat = est.predict(Xte_ft.loc[mask_te])
        yhat = pd.Series(yhat, index=yte_sub.index, name="y_pred")

        fold_preds = pd.DataFrame({
            "y_true": yte_sub,
            "y_pred": yhat,
            "train_end": pd.Timestamp(Xtr.index[-1]),
            "fold": fold_no,
        })
        preds.append(fold_preds)

    if not preds:
        return pd.DataFrame(), pd.DataFrame()

    preds_df = pd.concat(preds).sort_index()

    # Aggregierte Metriken
    metrics = {
        metric: scorer(preds_df["y_true"], preds_df["y_pred"]),
        "n_obs": int(len(preds_df)),
        "n_folds": fold_no,
    }
    metrics_df = pd.DataFrame([metrics])

    return preds_df, metrics_df
