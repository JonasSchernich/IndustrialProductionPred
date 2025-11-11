# src/models/ensemble.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import trim_mean

# Repo-abhängige Imports (Fallbacks erlauben Standalone-Betrieb)
try:
    from ..config import OUTPUTS  # type: ignore
    from ..evaluation import rmse  # type: ignore
except Exception:
    OUTPUTS = Path("outputs")

    def rmse(y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------
# Dataklassen & Utilities
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EnsembleConfig:
    """Konfiguration für Ensemble-Training/Anwendung."""
    stageA_dir: Path = OUTPUTS / "stageA"
    stageB_dir: Path = OUTPUTS / "stageB"
    save_dir: Path = OUTPUTS / "stageB" / "ensemble"   # wohin Stage-B-Ensemble-Outputs geschrieben werden
    ensure_dirs: bool = True


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_inner_join_on_index(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Inner-join über Index. Leere Ergebnis-Mengen früh erkennen."""
    if not dfs:
        raise ValueError("Leere DF-Liste zum Join erhalten.")
    out = dfs[0].copy()
    for df in dfs[1:]:
        out = out.join(df, how="inner")
    if out.empty:
        raise ValueError("Inner-Join ergab leeren Schnitt. Prüfe Zeitachsen und Modell-Läufe.")
    return out


# ---------------------------------------------------------------------
# Laden der Level-1 Daten (Stage A / Stage B) – neue Speicherlogik
# ---------------------------------------------------------------------

def _read_stageB_preds_for_model(
    model_tag: str,           # z.B. "lgbm_setup_II"
    stageB_base: Path,
    *,
    date_col: str = "date_t_plus_1",
    pred_col: str = "y_pred",
    use_only_is_active: bool = True,
) -> pd.DataFrame:
    """
    Erwartet: outputs/stageB/<model_tag>/monthly/preds.csv
    Nimmt (optional) nur is_active=True Zeilen.
    Gibt DF mit Index=date_col, Spalte=f"y_pred_{model_tag}" zurück.
    """
    f = stageB_base / model_tag / "monthly" / "preds.csv"
    if not f.exists():
        raise FileNotFoundError(f"Stage-B preds nicht gefunden: {f}")
    df = pd.read_csv(f, parse_dates=[date_col])
    if use_only_is_active and "is_active" in df.columns:
        df = df[df["is_active"] == True].copy()  # noqa: E712
    if pred_col not in df.columns:
        raise KeyError(f"Spalte '{pred_col}' fehlt in {f}")
    cur = (
        df[[date_col, pred_col]]
        .rename(columns={pred_col: f"y_pred_{model_tag}"})
        .set_index(date_col)
        .sort_index()
    )
    return cur


def _read_stageA_preds_for_model(
    model_tag: str,           # z.B. "lgbm_setup_II"
    stageA_base: Path,
    *,
    date_col: str = "date_t_plus_1",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """
    Erwartet: outputs/stageA/<model_tag>/block*/preds.csv
    Liest ALLE Blöcke und konkateniert deren OOS-Preds.
    Gibt DF mit Index=date_col, Spalte=f"y_pred_{model_tag}" zurück.
    """
    model_dir = stageA_base / model_tag
    if not model_dir.exists():
        raise FileNotFoundError(f"Stage-A Modellverzeichnis fehlt: {model_dir}")

    block_dirs = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.lower().startswith("block")]
    )
    if not block_dirs:
        raise FileNotFoundError(f"Keine block*-Ordner unter {model_dir} gefunden.")

    frames: List[pd.DataFrame] = []
    for bdir in block_dirs:
        f = bdir / "preds.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=[date_col])
        if pred_col not in df.columns:
            raise KeyError(f"Spalte '{pred_col}' fehlt in {f}")
        cur = (
            df[[date_col, pred_col]]
            .rename(columns={pred_col: f"y_pred_{model_tag}"})
            .set_index(date_col)
            .sort_index()
        )
        frames.append(cur)

    if not frames:
        raise FileNotFoundError(
            f"Keine Stage-A preds.csv in Blöcken gefunden für {model_tag} unter {model_dir}"
        )

    out = pd.concat(frames, axis=0).sort_index()
    # doppelte Indizes vermeiden (erste Beobachtung behalten)
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_level1_data(
    model_tags: List[str],            # z.B. ["elastic_net_setup_I", "svr_setup_I", ...]
    base_dir: Path,                   # i.d.R. OUTPUTS
    y_true_series: pd.Series,
    *,
    stage: str,                       # "A" oder "B"
    use_only_is_active_B: bool = True,
    date_col: str = "date_t_plus_1",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """
    NEUE Speicherlogik:

        Stage B:
            outputs/stageB/<model_tag>/monthly/preds.csv

        Stage A:
            outputs/stageA/<model_tag>/block*/preds.csv

    Rückgabe-DF:
        Index  : DatetimeIndex (Schnittmenge über alle Modelle + y_true)
        Spalten: 'y_true', 'y_pred_<model_tag_1>', 'y_pred_<model_tag_2>', ...
    """
    base = y_true_series.rename("y_true").to_frame()
    pieces: List[pd.DataFrame] = []
    missing: List[str] = []

    stage = stage.upper().strip()
    if stage not in {"A", "B"}:
        raise ValueError("Argument 'stage' muss 'A' oder 'B' sein.")

    stageA_base = base_dir / "stageA"
    stageB_base = base_dir / "stageB"

    for tag in model_tags:
        try:
            if stage == "B":
                cur = _read_stageB_preds_for_model(
                    tag,
                    stageB_base,
                    date_col=date_col,
                    pred_col=pred_col,
                    use_only_is_active=use_only_is_active_B,
                )
            else:
                cur = _read_stageA_preds_for_model(
                    tag,
                    stageA_base,
                    date_col=date_col,
                    pred_col=pred_col,
                )
            pieces.append(cur)
        except FileNotFoundError:
            missing.append(tag)

    if missing:
        print(f"[Ensemble] Warnung: Für {len(missing)} Modelle keine preds gefunden: {missing}")
    if not pieces:
        raise FileNotFoundError("Keine preds.csv-Dateien für die angegebenen Modelle gefunden.")

    out = _safe_inner_join_on_index([base] + pieces)
    return out


def get_pred_cols(df_l1: pd.DataFrame) -> List[str]:
    return [c for c in df_l1.columns if c.startswith("y_pred_")]


# ---------------------------------------------------------------------
# Ensemble-Strategien
# ---------------------------------------------------------------------

def apply_equal_weight(df_l1: pd.DataFrame, pred_cols: List[str]) -> pd.Series:
    """Equal-weight Durchschnitt (keine Parameter, leakage-sicher)."""
    return df_l1[pred_cols].mean(axis=1)


def apply_trimmed_mean(
    df_l1: pd.DataFrame,
    pred_cols: List[str],
    *,
    use_median: bool = False,
    trim_each_side: float = 0.0,
) -> pd.Series:
    """
    Symmetrisch getrimmtes Mittel oder Median.

    Parameter:
    - use_median: True => exakt der Median (robusteste Variante).
    - trim_each_side ∈ [0, 0.5): Anteil je Seite, der abgeschnitten wird (z.B. 0.1 => 10% pro Seite).
      Intern nutzt scipy.stats.trim_mean(proportiontocut=trim_each_side).

    Hinweise:
    - 'trim_each_side = 0.0' entspricht einfachem Mittel (wenn use_median=False).
    - Median kann NICHT mit trim_mean exakt dargestellt werden; deshalb Sonderfall.
    """
    if use_median:
        return df_l1[pred_cols].median(axis=1)

    if not (0.0 <= trim_each_side < 0.5):
        raise ValueError("trim_each_side muss in [0.0, 0.5) liegen.")

    return df_l1[pred_cols].apply(
        lambda row: trim_mean(row.values, proportiontocut=trim_each_side),
        axis=1,
    )


def compute_bates_granger_weights(
    df_l1: pd.DataFrame,
    pred_cols: List[str],
    *,
    window: int,
    w_min: float = 0.0,
    w_max: float = 0.6,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Berechne rollierende inverse-MSFE-Gewichte (Bates-Granger), ohne Leakage:
    - Rolling MSFE über Vergangenheits-Fehler (MSE der einzelnen Modelle)
    - inverse Gewichte, mit Cap/Floor pro Zeitpunkt
    - Normalisierung pro Zeitpunkt
    - Ergebnis: Gewichte, die zu ZEIT t auf Prognose t+1 angewandt werden müssen -> daher noch .shift(1)!

    Rückgabe: DataFrame 'weights' (gleicher Index wie df_l1), Spalten = pred_cols (unge-shiftet).
    """
    # Squared errors (pro Modell vs. y_true)
    se = df_l1[pred_cols].subtract(df_l1["y_true"], axis=0).pow(2)

    # Rolling MSFE (stabiler Start)
    msfe = se.rolling(window=window, min_periods=max(2, window // 2)).mean()

    inv = 1.0 / (msfe + eps)
    w = inv.div(inv.sum(axis=1), axis=0)

    # Cap/Floor -> erneut normalisieren (robust gegen Zeilen mit Summe=0)
    w = w.clip(lower=w_min, upper=w_max)
    denom = w.sum(axis=1).replace(0, np.nan)
    # divide row-wise by denom (axis=0 aligns on index)
    w = w.div(denom, axis=0)
    # Falls Zeilen NaNs haben (z.B. denom=0): auf Equal Weights zurückfallen
    if w.isna().any().any():
        w = w.apply(
            lambda row: row.fillna(1.0 / len(pred_cols)),
            axis=1,
        )
    return w


def apply_bates_granger(
    df_l1: pd.DataFrame,
    pred_cols: List[str],
    *,
    window: int,
    w_min: float = 0.0,
    w_max: float = 0.6,
) -> pd.Series:
    """
    Bates-Granger-Ensemble-Vorhersagen (inverse MSFE, leakage-sicher).
    """
    M = len(pred_cols)
    w = compute_bates_granger_weights(
        df_l1,
        pred_cols,
        window=window,
        w_min=w_min,
        w_max=w_max,
    )
    w_shift = w.shift(1).fillna(1.0 / M)
    yhat = (df_l1[pred_cols] * w_shift).sum(axis=1)
    yhat.name = "ENS_BG"
    return yhat


def _rolling_mad_scale(y: np.ndarray, window: int) -> np.ndarray:
    """
    Leakage-freie, robuste Skala via rollierender MAD (1.4826 * median|x - median(x)|),
    nur Vergangenheit (ffill), untere Schwelle.
    """
    s = pd.Series(y)
    mad = s.rolling(window=window, min_periods=1).apply(
        lambda x: 1.4826 * np.median(np.abs(x - np.median(x))),
        raw=False,
    )
    mad = mad.replace(0, np.nan).ffill()
    if pd.isna(mad.iloc[0]):
        mad.iloc[0] = 1.0
    mad = mad.ffill().clip(lower=1e-6)
    return mad.to_numpy()


def compute_ewa_ensemble(
    df_l1: pd.DataFrame,
    pred_cols: List[str],
    *,
    eta: float,
    delta: float = 1.0,
    scale_window: int = 24,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    EWA/Hedge: sequentielle Gewichte aus skaliertem, ggf. vergessendem Verlust.

    Rückgabe:
      - y_hat_ens (Series)    : Ensemble-Prognose (leakage-sicher)
      - weights_hist (DataFrame): Gewichte je t (die auf Prognose t angewandt wurden)
    """
    preds = df_l1[pred_cols].to_numpy()
    y = df_l1["y_true"].to_numpy()
    T, M = preds.shape

    # Start
    weights = np.ones(M) / M
    weights_hist = np.zeros((T, M))
    y_hat = np.zeros(T)

    # leakage-freie Skala: Skala von t-1 verwenden
    scale_raw = _rolling_mad_scale(y, scale_window)
    scale_shift = pd.Series(scale_raw).shift(1).ffill().fillna(1.0).to_numpy()

    cum_loss = np.zeros(M)
    for t in range(T):
        # Prognose mit Gewichten von t (die aus Verl. bis t-1 stammen)
        y_hat[t] = float(np.dot(weights, preds[t, :]))
        weights_hist[t, :] = weights

        # Beobachtete Verluste zum Update für t+1
        losses_raw = (y[t] - preds[t, :]) ** 2
        s = scale_shift[t]
        losses_scaled = np.clip(losses_raw / (s**2 + 1e-9), 0.0, 1.0)
        cum_loss = delta * cum_loss + losses_scaled

        # Update (stabilisiert)
        logw = -eta * cum_loss
        logw -= logw.min()
        w_unnorm = np.exp(logw)
        weights = w_unnorm / (w_unnorm.sum() + 1e-12)

    wdf = pd.DataFrame(weights_hist, index=df_l1.index, columns=pred_cols)
    yhat = pd.Series(y_hat, index=df_l1.index, name="ENS_EWA")
    return yhat, wdf


def apply_ewa(
    df_l1: pd.DataFrame,
    pred_cols: List[str],
    *,
    eta: float,
    delta: float = 1.0,
    scale_window: int = 24,
) -> pd.Series:
    yhat, _ = compute_ewa_ensemble(
        df_l1,
        pred_cols,
        eta=eta,
        delta=delta,
        scale_window=scale_window,
    )
    return yhat


# ---------------------------------------------------------------------
# Stacking (Stage-A fit, Stage-B frozen)
# ---------------------------------------------------------------------

def tune_stacking_weights(
    df_tune: pd.DataFrame,
    pred_cols: List[str],
    *,
    ridge_lambda: float = 0.1,
    start_from_best: bool = True,
) -> np.ndarray:
    """
    Convex, nonnegative Stacking-Gewichte (Summe=1) mit Ridge-Shrinkage zum EW-Prior.
    Training nur auf Stage-A (echte OOS), Gewichte werden für Stage-B eingefroren.

    Ziel:
        min_w  mean((y - Xw)^2) + λ * ||w - (1/M)||^2
        s.t.   w_i >= 0, sum_i w_i = 1

    Rückgabe: np.ndarray (M,)
    """
    X = df_tune[pred_cols].to_numpy()
    y = df_tune["y_true"].to_numpy()
    M = X.shape[1]
    prior = np.ones(M) / M

    # Startpunkt
    if start_from_best:
        rmses = [rmse(y, X[:, i]) for i in range(M)]
        w0 = np.zeros(M)
        w0[int(np.argmin(rmses))] = 1.0
    else:
        w0 = prior.copy()

    def objective(w: np.ndarray) -> float:
        yhat = X @ w
        loss = np.mean((y - yhat) ** 2)
        penalty = ridge_lambda * np.sum((w - prior) ** 2)
        return float(loss + penalty)

    bounds = tuple((0.0, 1.0) for _ in range(M))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not res.success:
        print(f"[Ensemble] Warnung: Stacking-Optimierung nicht konvergiert: {res.message}. Fallback=EW.")
        return prior

    w = res.x
    w = w / (w.sum() + 1e-12)
    return w


def apply_stacking(
    df_eval: pd.DataFrame,
    pred_cols: List[str],
    weights: np.ndarray,
) -> pd.Series:
    """Anwendung eingefrorener Stacking-Gewichte (ohne Nachlernen)."""
    yhat = df_eval[pred_cols].to_numpy() @ np.asarray(weights)
    return pd.Series(yhat, index=df_eval.index, name="ENS_Stacking")


# ---------------------------------------------------------------------
# Speichern & Reporting
# ---------------------------------------------------------------------

def save_ensemble_predictions(
    y_true: pd.Series,
    ens_preds: Dict[str, pd.Series],
    *,
    save_dir: Path,
    run_name: str,
) -> None:
    """
    Speichert Ensemble-Serien im Format ähnlich der Basismodelle:
    save_dir/<run_name>/monthly/preds__<ensemble_name>.csv
    mit Spalten: date_t_plus_1, y_pred, is_active, ensemble_name
    """
    out_dir = Path(save_dir) / run_name / "monthly"
    _ensure_dir(out_dir)

    for name, s in ens_preds.items():
        df = pd.DataFrame(
            {
                "date_t_plus_1": s.index,
                "y_pred": s.values,
                "is_active": True,
                "ensemble_name": name,
            }
        )
        df.to_csv(out_dir / f"preds__{name}.csv", index=False)

    # Optional: ein kombiniertes File
    comb = pd.DataFrame({"date_t_plus_1": y_true.index, "y_true": y_true.values})
    for name, s in ens_preds.items():
        comb = comb.merge(
            s.rename(name).rename_axis("date_t_plus_1").reset_index(),
            on="date_t_plus_1",
            how="left",
        )
    comb.to_csv(out_dir / "preds__ALL.csv", index=False)


def rmse_table(
    y_true: pd.Series,
    df_or_dict: Dict[str, pd.Series] | pd.DataFrame
) -> pd.DataFrame:
    """
    Erzeugt eine RMSE-Tabelle aus einer Mapping-Struktur oder einem DataFrame.
    """
    if isinstance(df_or_dict, dict):
        cols = {}
        for k, s in df_or_dict.items():
            cols[k] = float(rmse(y_true.reindex(s.index), s))
        out = pd.Series(cols).sort_values().to_frame("RMSE")
        out.index.name = "Model"
        return out

    # DataFrame
    out = {}
    for col in df_or_dict.columns:
        if col == "y_true":
            continue
        out[col] = float(rmse(df_or_dict["y_true"], df_or_dict[col]))
    out = pd.Series(out).sort_values().to_frame("RMSE")
    out.index.name = "Model"
    return out

