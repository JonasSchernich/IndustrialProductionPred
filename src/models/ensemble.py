from __future__ import annotations

"""
Ensemble utilities for combining Level-0 nowcasts from heterogeneous models
"""

from dataclasses import dataclass
from typing import Sequence, List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Absolute imports so this works from notebooks and scripts
from src.config import outputs_for_model
from src.evaluation import rmse


# ---------------------------------------------------------------------------
# I/O: Load Level-0 OOS predictions
# ---------------------------------------------------------------------------

@dataclass
class Level0Pool:
    """Aligned Level-0 OOS predictions."""
    dates: pd.DatetimeIndex
    y_true: pd.Series
    F: pd.DataFrame
    model_names: List[str]


def _load_active_stageB_predictions(model_name: str) -> pd.DataFrame:
    """Load Stage B OOS predictions for a model, keeping only the active configuration."""
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

    df["date_t_plus_1"] = pd.to_datetime(df["date_t_plus_1"])
    df = df.sort_values("date_t_plus_1").set_index("date_t_plus_1")

    # Drop duplicates (keep last)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    return df[["y_true", "y_pred"]].rename(columns={"y_pred": model_name})


def _load_stageA_champion_predictions(model_name: str) -> pd.DataFrame:
    """Load Stage A 'champion' predictions (best RMSE in block3) for calibration."""
    outs = outputs_for_model(model_name)
    out_dir = outs["stageA"]

    # Find champion config from block3 summary
    block3_dir = out_dir / "block3"
    summary_path = block3_dir / "rmse.csv"

    if not summary_path.exists():
        # Fallback: try summary.csv in the main output folder
        summary_path = out_dir / "summary" / "summary.csv"

    if not summary_path.exists():
        print(f"WARNING: No Stage A data found for {model_name}. Calibration data is missing.")
        return pd.DataFrame()

    df_sum = pd.read_csv(summary_path)

    # Column name may vary
    rmse_col = "rmse" if "rmse" in df_sum.columns else "rmse_val"

    best_cfg_row = df_sum.sort_values(rmse_col).iloc[0]
    best_cfg_id = best_cfg_row["config_id"]

    # Load predictions for that config
    preds_path = block3_dir / "preds.csv"
    if not preds_path.exists():
        return pd.DataFrame()

    df_preds = pd.read_csv(preds_path)
    df_preds["date_t_plus_1"] = pd.to_datetime(df_preds["date_t_plus_1"])

    champ_preds = df_preds[df_preds["config_id"] == best_cfg_id].copy()
    champ_preds = champ_preds.sort_values("date_t_plus_1").set_index("date_t_plus_1")

    # Drop duplicates (keep last), e.g. after re-runs
    if champ_preds.index.duplicated().any():
        champ_preds = champ_preds[~champ_preds.index.duplicated(keep="last")]

    return champ_preds[["y_true", "y_pred"]].rename(columns={"y_pred": model_name})


def load_level0_pool(model_names: Sequence[str]) -> Level0Pool:
    """Load Stage B predictions."""
    model_names = list(model_names)
    dfs: Dict[str, pd.DataFrame] = {}
    for name in model_names:
        dfs[name] = _load_active_stageB_predictions(name)
    return _align_and_build_pool(dfs, model_names)


def load_calibration_pool(model_names: Sequence[str]) -> Level0Pool:
    """Load Stage A predictions (for calibration)."""
    model_names = list(model_names)
    dfs: Dict[str, pd.DataFrame] = {}
    for name in model_names:
        df = _load_stageA_champion_predictions(name)
        if not df.empty:
            dfs[name] = df

    if not dfs:
        raise ValueError("Could not load any Stage A calibration data.")

    return _align_and_build_pool(dfs, model_names)


def _align_and_build_pool(dfs: Dict[str, pd.DataFrame], model_names: List[str]) -> Level0Pool:
    # Align on common dates
    common_idx = None
    for df in dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common dates across base models.")

    common_idx = common_idx.sort_values()

    # Take y_true from the first model (assumed consistent across models)
    first_name = list(dfs.keys())[0]
    y_true = dfs[first_name].loc[common_idx, "y_true"]
    y_true.name = "y_true"

    # Build forecast matrix
    F = pd.DataFrame(index=common_idx)
    for name in model_names:
        if name in dfs:
            F[name] = dfs[name].loc[common_idx, name].astype(float)
        else:
            # Should not happen (kept for robustness)
            F[name] = np.nan

    return Level0Pool(dates=common_idx, y_true=y_true, F=F, model_names=model_names)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def equal_weight_ensemble(F: pd.DataFrame) -> pd.Series:
    return F.mean(axis=1)


def trimmed_mean_ensemble(F: pd.DataFrame, alpha: float) -> pd.Series:
    if not (0.0 <= alpha < 0.5):
        raise ValueError("alpha must be in [0, 0.5).")
    M = F.shape[1]
    k = int(np.floor(alpha * M))
    if k == 0:
        return equal_weight_ensemble(F)

    def _trimmed_row(x: np.ndarray) -> float:
        xs = np.sort(x.astype(float))
        if 2 * k >= len(xs):
            return float(np.median(xs))
        return float(xs[k:-k].mean())

    vals = np.apply_along_axis(_trimmed_row, 1, F.values)
    return pd.Series(vals, index=F.index, name="trimmed_mean")


def median_ensemble(F: pd.DataFrame) -> pd.Series:
    return F.median(axis=1)


@dataclass
class StackingResult:
    weights: pd.Series
    lambda_opt: float
    rmse_cal: float
    y_pred: pd.Series


def fit_stacking_ensemble(
    y: pd.Series,
    F: pd.DataFrame,
    cal_dates: Sequence[pd.Timestamp],
    lambdas: Sequence[float],
) -> StackingResult:
    if len(F.columns) == 0:
        raise ValueError("F empty")

    cal_idx = pd.Index(cal_dates).intersection(F.index).sort_values()
    if len(cal_idx) == 0:
        raise ValueError("No overlap between cal_dates and F.index")

    y_cal = y.loc[cal_idx].values
    F_cal = F.loc[cal_idx].values
    T, M = F_cal.shape
    w_equal = np.full(M, 1.0 / M)

    bounds = [(0.0, None)] * M
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def make_obj(lam: float):
        def _obj(w):
            resid = y_cal - F_cal @ w
            reg = lam * np.sum((w - w_equal) ** 2)
            return np.dot(resid, resid) + reg

        return _obj

    best_res = None
    best_rmse = np.inf

    for lam in lambdas:
        res = minimize(make_obj(lam), w_equal, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w_hat = res.x
            w_hat[w_hat < 0] = 0
            w_hat /= w_hat.sum()

            this_rmse = rmse(y_cal, F_cal @ w_hat)
            if this_rmse < best_rmse:
                best_rmse = this_rmse
                best_res = (w_hat, lam)

    if best_res is None:
        w_final, lam_final = w_equal, lambdas[0]
    else:
        w_final, lam_final = best_res

    y_pred_all = pd.Series(F.values @ w_final, index=F.index, name="stacked")
    w_series = pd.Series(w_final, index=F.columns, name="weights")

    return StackingResult(w_series, lam_final, best_rmse, y_pred_all)


@dataclass
class EWAResult:
    eta_opt: float
    delta: float
    rmse_cal: float
    y_pred: pd.Series
    weights_history: pd.DataFrame


def _run_ewa_single(y, F, eta, delta=1.0):
    dates = F.index.intersection(y.index).sort_values()
    y_v = y.loc[dates].values
    F_v = F.loc[dates].values
    T, M = F_v.shape

    L = np.zeros(M)
    weights_out = []
    preds_out = []

    # Online variance stats (Welford)
    n_seen = 0
    mean_y = 0.0
    m2_y = 0.0

    for t in range(T):
        exps = -eta * L
        exps -= exps.max()
        w = np.exp(exps)
        w /= w.sum()
        weights_out.append(w)

        f_t = F_v[t]
        y_hat = np.dot(w, f_t)
        preds_out.append(y_hat)

        y_t = y_v[t]

        n_seen += 1
        delta_val = y_t - mean_y
        mean_y += delta_val / n_seen
        delta2 = y_t - mean_y
        m2_y += delta_val * delta2

        if n_seen < 2:
            s2 = max((y_t - mean_y) ** 2, 1e-6)
        else:
            s2 = max(m2_y / (n_seen - 1), 1e-6)

        loss_t = (y_t - f_t) ** 2 / s2
        loss_t = np.clip(loss_t, 0, 1)
        L = delta * L + loss_t

    return (
        pd.Series(preds_out, index=dates, name="ewa"),
        pd.DataFrame(weights_out, index=dates, columns=F.columns),
    )


def fit_ewa_ensemble(y, F, cal_dates, etas, delta=0.95):
    cal_idx = pd.Index(cal_dates).intersection(F.index)

    best_eta = etas[0]
    best_rmse = np.inf

    # Tune on calibration window
    for eta in etas:
        y_hat, _ = _run_ewa_single(y.loc[cal_idx], F.loc[cal_idx], eta, delta)
        r = rmse(y.loc[cal_idx].values, y_hat.values)
        if r < best_rmse:
            best_rmse = r
            best_eta = eta

    y_full, W_full = _run_ewa_single(y, F, best_eta, delta)
    return EWAResult(best_eta, delta, best_rmse, y_full, W_full)
