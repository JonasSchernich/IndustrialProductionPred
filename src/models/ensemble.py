# src/ensemble.py
from __future__ import annotations

"""
Ensemble utilities for combining Level-0 nowcasts from heterogeneous models.

This module is fully leakage-safe: all weights are estimated using *out-of-sample*
prequential predictions from the base models only, and only using information
that would have been available at the time.

We implement three combination schemes, matching the thesis text:

1. Equal-weight (and trimmed) means.
2. Static stacked regression with convex weights and ridge shrinkage
   toward equal weights.
3. Online Exponentially Weighted Averaging (EWA / Hedge) with optional
   exponential discounting of past losses.

All routines operate on pre-computed OOS predictions stored in
`outputs/stageB/<model_name>/monthly/preds.csv` and do *not* refit the
base models.
"""

from dataclasses import dataclass
from typing import Sequence, List, Dict, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from .config import outputs_for_model
from .evaluation import rmse


# ---------------------------------------------------------------------------
# I/O: Load Level-0 OOS predictions
# ---------------------------------------------------------------------------

@dataclass
class Level0Pool:
    """Container for aligned Level-0 OOS predictions.

    Attributes
    ----------
    dates : pd.DatetimeIndex
        OOS dates (t+1 in thesis notation).
    y_true : pd.Series
        Realized targets y_{t+1} aligned with `dates`.
    F : pd.DataFrame
        Matrix of base-model forecasts with shape (T_oos, M),
        columns labelled by `model_names`.
    model_names : List[str]
        Names of the base models.
    """
    dates: pd.DatetimeIndex
    y_true: pd.Series
    F: pd.DataFrame
    model_names: List[str]


def _load_active_stageB_predictions(model_name: str) -> pd.DataFrame:
    """
    Load OOS predictions from Stage B for a given model, keeping only the
    *active* configuration chosen by the internal online policy.

    Returns a dataframe indexed by date_t_plus_1 with columns:
        - y_true
        - <model_name> (nowcast)
    """
    outs = outputs_for_model(model_name)
    preds_path = outs["stageB"] / "monthly" / "preds.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Stage B preds not found for model '{model_name}': {preds_path}")

    df = pd.read_csv(preds_path)
    if "date_t_plus_1" not in df.columns:
        raise ValueError(f"'date_t_plus_1' column missing in {preds_path}")

    # Keep only active configuration per month
    if "is_active" in df.columns:
        df = df[df["is_active"] == True].copy()

    # Parse date and set index
    df["date_t_plus_1"] = pd.to_datetime(df["date_t_plus_1"])
    df = df.sort_values("date_t_plus_1")
    df = df.set_index("date_t_plus_1")

    # Basic sanity check
    if df.index.duplicated().any():
        # In case of duplicates, keep the last prediction per date
        df = df[~df.index.duplicated(keep="last")]

    return df[["y_true", "y_pred"]].rename(columns={"y_pred": model_name})


def load_level0_pool(model_names: Sequence[str]) -> Level0Pool:
    """
    Load and align Level-0 OOS predictions for a list of models.

    Parameters
    ----------
    model_names : sequence of str
        Model names as used in the Stage B runs (e.g. 'elastic_net', 'lightgbm', ...).

    Returns
    -------
    Level0Pool
        Aligned target and forecast matrix across all base models.
    """
    model_names = list(model_names)
    if len(model_names) == 0:
        raise ValueError("At least one model_name is required.")

    dfs: Dict[str, pd.DataFrame] = {}
    for name in model_names:
        df = _load_active_stageB_predictions(name)
        dfs[name] = df

    # Align on common dates
    common_idx = None
    for df in dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common dates across base models.")

    common_idx = common_idx.sort_values()

    # Build y_true (sanity check equality across models)
    y_true = None
    for name, df in dfs.items():
        y_part = df.loc[common_idx, "y_true"]
        if y_true is None:
            y_true = y_part.copy()
        else:
            if not np.allclose(y_true.values, y_part.values, atol=1e-8, equal_nan=True):
                raise ValueError(f"y_true mismatch between models; check outputs for '{name}'.")

    assert y_true is not None
    y_true.name = "y_true"

    # Build forecast matrix
    F = pd.DataFrame(index=common_idx)
    for name, df in dfs.items():
        F[name] = df.loc[common_idx, name].astype(float)

    return Level0Pool(dates=common_idx, y_true=y_true, F=F, model_names=model_names)


# ---------------------------------------------------------------------------
# Equal-weight and trimmed means
# ---------------------------------------------------------------------------

def equal_weight_ensemble(F: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional equal-weight average of forecasts.

    Parameters
    ----------
    F : DataFrame (T x M)
        Rows = dates, columns = base models.

    Returns
    -------
    pd.Series
        Ensemble prediction per date.
    """
    return F.mean(axis=1)


def trimmed_mean_ensemble(F: pd.DataFrame, alpha: float) -> pd.Series:
    """
    Symmetric alpha-trimmed mean over base-model forecasts.

    Parameters
    ----------
    F : DataFrame (T x M)
        Rows = dates, columns = base models.
    alpha : float
        Fraction in (0, 0.5). A value of 0.1 drops the lowest 10% and highest
        10% of forecasts (rounded down) before averaging.

    Returns
    -------
    pd.Series
        Ensemble prediction per date.
    """
    if not (0.0 <= alpha < 0.5):
        raise ValueError("alpha must be in [0, 0.5).")

    M = F.shape[1]
    k = int(np.floor(alpha * M))
    if k == 0:
        # No trimming → equal-weight
        return equal_weight_ensemble(F)

    def _trimmed_row(x: np.ndarray) -> float:
        xs = np.sort(x.astype(float))
        if 2 * k >= len(xs):
            # degenerate case: fall back to median
            return float(np.median(xs))
        return float(xs[k:-k].mean())

    vals = np.apply_along_axis(_trimmed_row, 1, F.values)
    return pd.Series(vals, index=F.index, name="trimmed_mean")


def median_ensemble(F: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional median of base-model forecasts.
    """
    return F.median(axis=1)


# ---------------------------------------------------------------------------
# Stacked regression (static meta-learner)
# ---------------------------------------------------------------------------

@dataclass
class StackingResult:
    weights: pd.Series       # convex weights over base models
    lambda_opt: float        # selected ridge penalty
    rmse_cal: float          # RMSE on calibration period
    y_pred: pd.Series        # stacked predictions over *all* dates


def fit_stacking_ensemble(
    y: pd.Series,
    F: pd.DataFrame,
    cal_dates: Sequence[pd.Timestamp],
    lambdas: Sequence[float],
) -> StackingResult:
    """
    Fit convex stacked regression with ridge shrinkage toward equal weights.

    The problem is
        min_w  sum_{tau in cal} (y_tau - F_tau w)^2
              + lambda ||w - 1/M||_2^2
        s.t.   w_i >= 0, sum_i w_i = 1.

    Parameters
    ----------
    y : pd.Series
        Realized targets, indexed by date.
    F : pd.DataFrame
        Base-model forecasts, index aligned with `y`.
    cal_dates : sequence of T_cal dates
        Calibration dates (subset of F.index) used to estimate weights.
    lambdas : sequence of float
        Candidate ridge penalties.

    Returns
    -------
    StackingResult
    """
    if len(F.columns) == 0:
        raise ValueError("F must have at least one column.")
    if len(cal_dates) == 0:
        raise ValueError("cal_dates must be non-empty.")

    lambdas = list(lambdas)
    if len(lambdas) == 0:
        raise ValueError("At least one lambda must be provided.")

    # Restrict to calibration period
    cal_idx = pd.Index(cal_dates)
    cal_idx = cal_idx.intersection(F.index).sort_values()
    if len(cal_idx) == 0:
        raise ValueError("No overlap between cal_dates and forecast index.")

    y_cal = y.loc[cal_idx].astype(float).values
    F_cal = F.loc[cal_idx].astype(float).values

    T_cal, M = F_cal.shape
    w_equal = np.full(M, 1.0 / M, dtype=float)

    if T_cal < M:
        # Underdetermined: fall back to equal weights
        w_best = w_equal
        lambda_best = float(lambdas[0])
        rmse_best = rmse(y_cal, F_cal @ w_best)
    else:
        bounds = [(0.0, None)] * M
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]

        def make_obj(lam: float):
            def _obj(w: np.ndarray) -> float:
                w = np.asarray(w, dtype=float)
                resid = y_cal - F_cal @ w
                reg = lam * float(np.sum((w - w_equal) ** 2))
                return float(np.dot(resid, resid) + reg)
            return _obj

        w0 = w_equal.copy()
        lambda_best = None
        w_best = None
        rmse_best = np.inf

        for lam in lambdas:
            obj = make_obj(float(lam))
            try:
                res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
                if not res.success:
                    continue
                w_hat = np.asarray(res.x, dtype=float)
                # numerical clean-up
                w_hat[w_hat < 0] = 0.0
                s = float(w_hat.sum())
                if s <= 0:
                    w_hat = w_equal.copy()
                else:
                    w_hat /= s
            except Exception:
                # fall back to equal weights if optimizer fails
                w_hat = w_equal.copy()

            y_cal_hat = F_cal @ w_hat
            rmse_lam = rmse(y_cal, y_cal_hat)
            if rmse_lam < rmse_best:
                rmse_best = rmse_lam
                lambda_best = float(lam)
                w_best = w_hat.copy()

        if w_best is None:
            w_best = w_equal
            lambda_best = float(lambdas[0])
            rmse_best = rmse(y_cal, F_cal @ w_best)

    # Apply static weights to full sample
    F_all = F.astype(float).values
    y_hat_all = F_all @ w_best
    y_hat_all = pd.Series(y_hat_all, index=F.index, name="stacked")

    weights = pd.Series(w_best, index=F.columns, name="stack_weights")

    return StackingResult(weights=weights, lambda_opt=lambda_best, rmse_cal=rmse_best, y_pred=y_hat_all)


# ---------------------------------------------------------------------------
# Exponentially Weighted Averaging (EWA / Hedge)
# ---------------------------------------------------------------------------

@dataclass
class EWAResult:
    eta_opt: float                # selected learning rate
    delta: float                  # forgetting factor
    rmse_cal: float               # RMSE on calibration period
    y_pred: pd.Series             # EWA predictions over all dates
    weights_history: pd.DataFrame # per-date weights over base models


def _run_ewa_single(
    y: pd.Series,
    F: pd.DataFrame,
    eta: float,
    delta: float = 1.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run EWA/Hedge on a given series y and forecast matrix F.

    Parameters
    ----------
    y : pd.Series
        Realized targets, indexed by date.
    F : pd.DataFrame
        Base-model forecasts (aligned with y).
    eta : float
        Learning rate > 0.
    delta : float, optional
        Forgetting factor in (0, 1]; delta=1.0 corresponds to no discounting.

    Returns
    -------
    (y_hat, W)
        y_hat : pd.Series of ensemble predictions per date.
        W     : pd.DataFrame of weights per date (rows=dates, cols=base models).
    """
    if eta <= 0:
        raise ValueError("eta must be > 0.")
    if not (0.0 < delta <= 1.0):
        raise ValueError("delta must be in (0, 1].")

    dates = F.index.intersection(y.index)
    dates = dates.sort_values()
    F = F.loc[dates].astype(float)
    y = y.loc[dates].astype(float)

    M = F.shape[1]
    L = np.zeros(M, dtype=float)  # cumulative (discounted) losses

    # Running variance for scale s_t^2 (Welford)
    n = 0
    mean = 0.0
    m2 = 0.0

    weights_list: List[pd.Series] = []
    yhat_list: List[float] = []

    for date in dates:
        # 1) Compute weights based on past losses L
        exponents = -eta * L
        # numerical stabilisation
        exponents -= np.max(exponents)
        w = np.exp(exponents)
        s = float(w.sum())
        if s <= 0:
            w[:] = 1.0 / M
        else:
            w /= s

        w_series = pd.Series(w, index=F.columns, name=date)
        weights_list.append(w_series)

        # 2) Ensemble prediction
        f_t = F.loc[date].values
        y_hat_t = float(np.dot(w, f_t))
        yhat_list.append(y_hat_t)

        # 3) Observe outcome and update scale s_t^2 (only past and current y)
        y_t = float(y.loc[date])
        n += 1
        if n == 1:
            mean = y_t
            m2 = 0.0
            var = max((y_t - mean) ** 2, 1e-6)
        else:
            delta_y = y_t - mean
            mean += delta_y / n
            m2 += delta_y * (y_t - mean)
            var = m2 / max(n - 1, 1.0)
            var = max(var, 1e-6)

        s2 = var

        # 4) Per-expert loss
        se = (y_t - f_t) ** 2
        tilde_l = se / s2
        ell = np.clip(tilde_l, 0.0, 1.0)

        # 5) Discounted cumulative loss update
        L = delta * L + ell

    y_hat = pd.Series(yhat_list, index=dates, name="ewa")
    W = pd.DataFrame(weights_list)
    return y_hat, W


def fit_ewa_ensemble(
    y: pd.Series,
    F: pd.DataFrame,
    cal_dates: Sequence[pd.Timestamp],
    etas: Sequence[float],
    delta: float = 0.95,
) -> EWAResult:
    """
    Fit EWA by selecting the learning rate eta on a calibration period.

    Parameters
    ----------
    y : pd.Series
        Realized targets, indexed by date.
    F : pd.DataFrame
        Base-model forecasts (aligned with y).
    cal_dates : sequence of dates
        Calibration dates used to tune eta.
    etas : sequence of float
        Candidate learning rates (> 0).
    delta : float, optional
        Forgetting factor in (0, 1]; default 0.95 as in the thesis.

    Returns
    -------
    EWAResult
    """
    etas = list(etas)
    if len(etas) == 0:
        raise ValueError("At least one eta must be provided.")

    cal_idx = pd.Index(cal_dates)
    cal_idx = cal_idx.intersection(F.index).sort_values()
    if len(cal_idx) == 0:
        raise ValueError("No overlap between cal_dates and forecast index.")

    y_cal = y.loc[cal_idx]
    F_cal = F.loc[cal_idx]

    eta_best = None
    rmse_best = np.inf

    for eta in etas:
        y_hat_cal, _ = _run_ewa_single(y_cal, F_cal, eta=float(eta), delta=delta)
        r = rmse(y_cal.values, y_hat_cal.values)
        if r < rmse_best:
            rmse_best = r
            eta_best = float(eta)

    if eta_best is None:
        eta_best = float(etas[0])
        y_hat_cal, _ = _run_ewa_single(y_cal, F_cal, eta=eta_best, delta=delta)
        rmse_best = rmse(y_cal.values, y_hat_cal.values)

    # Run once more on full sample with optimal eta
    y_hat_all, W_all = _run_ewa_single(y, F, eta=eta_best, delta=delta)

    return EWAResult(
        eta_opt=eta_best,
        delta=delta,
        rmse_cal=rmse_best,
        y_pred=y_hat_all,
        weights_history=W_all,
    )
