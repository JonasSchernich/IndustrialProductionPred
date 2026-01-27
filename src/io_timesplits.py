# src/io_timesplits.py

from __future__ import annotations
from typing import Iterable, Iterator, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import os, re

# Relative imports
from .config import PROCESSED, OUTPUTS, outputs_for_model, GlobalConfig


def _validate_index(idx: pd.DatetimeIndex) -> None:
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Index must be strictly increasing.")


def load_target(col: str = "IP_change") -> pd.Series:
    """Load target.csv and return the requested target column (default: IP_change)."""
    path = PROCESSED / "target.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if col not in df.columns:
        raise ValueError(
            f"Spalte '{col}' nicht in target.csv gefunden. "
            f"Verfügbare Spalten: {list(df.columns)}"
        )
    s = df[col].astype(float).dropna().sort_index()
    _validate_index(s.index)
    return s


def _clean_column_name(col_name: str) -> str:
    """Make column names safe and predictable."""
    name = str(col_name)
    # Replace invalid chars with underscores
    name = re.sub(r"[^a-zA-Z0-9_.]", "_", name)
    # Prefix if name starts with a digit or period
    if re.match(r"^[0-9.]", name):
        name = "f_" + name
    # Fallback
    if not name:
        name = "unnamed_feature"
    return name


def load_ifo_features() -> pd.DataFrame:
    """Load cleaned_features.csv and ensure a valid DatetimeIndex."""
    path = PROCESSED / "cleaned_features.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    _validate_index(df.index)

    original_columns = df.columns.tolist()
    cleaned_columns = [_clean_column_name(col) for col in original_columns]

    if original_columns != cleaned_columns:
        print("INFO load_ifo_features: Renaming columns for validity.")
        df.columns = cleaned_columns

    return df


def _load_parquet_with_datetime_index(
    file_name: str,
    date_col: str = "date",
    base_dir: Path = PROCESSED,
) -> pd.DataFrame:
    """Load a parquet file and ensure a DatetimeIndex."""
    path = base_dir / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}. Bitte zuerst `feature_importance.ipynb` (oder andere Skripte) ausführen."
        )

    df = pd.read_parquet(path)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif df.index.name == date_col or df.index.name is None:
        # Already an index; ensure datetime and name it for reindexing.
        df.index = pd.to_datetime(df.index)
        df.index.name = date_col
    else:
        raise ValueError(f"Datumsspalte '{date_col}' nicht in {file_name} gefunden.")

    _validate_index(df.index)
    return df


def load_tsfresh() -> pd.DataFrame:
    """Load TSFresh features."""
    return _load_parquet_with_datetime_index("tsfresh_slim.parquet", date_col="date")


def load_chronos() -> pd.DataFrame:
    """Load Chronos features."""
    df = _load_parquet_with_datetime_index("chronos_bolt.parquet", date_col="date")
    return df


def load_ar() -> pd.DataFrame:
    """Load AR.parquet features."""
    df = _load_parquet_with_datetime_index("AR.parquet", date_col="date")
    df = df
    return df


def load_full_lagged_features(base_dir: Path | None = None) -> pd.DataFrame:
    """Load the full lagged feature matrix created by feature_importance.ipynb."""
    FI_DIR = base_dir or (OUTPUTS / "feature_importance")
    return _load_parquet_with_datetime_index(
        "X_eng_full_lagged.parquet",
        date_col="date",
        base_dir=FI_DIR,
    )


def load_rolling_importance(base_dir: Path | None = None) -> pd.DataFrame:
    """Load the rolling-mean importance matrix created by feature_importance.ipynb."""
    FI_DIR = base_dir or (OUTPUTS / "feature_importance")
    return _load_parquet_with_datetime_index(
        "rolling_mean_importance_60m.parquet",
        date_col="date",
        base_dir=FI_DIR,
    )


def stageA_blocks(cfg: GlobalConfig, T: int) -> Iterator[Tuple[int, int, int, int]]:
    """Yield (train_end, oos_start, oos_end, block_id); OOS bounds are inclusive."""
    blocks = list(cfg.BLOCKS_A)
    for i, (oos_start, oos_end) in enumerate(blocks, start=1):
        train_end = cfg.W0_A if i == 1 else (oos_start - 1)
        yield (train_end, oos_start, oos_end, i)


def stageB_months(cfg: GlobalConfig, T: int) -> Iterator[int]:
    """Yield origins t = cfg.W0_B .. T-1 (predict t+1)."""
    for t in range(cfg.W0_B, T):
        yield t


def ensure_outputs(model_name: str) -> Dict[str, Path]:
    return outputs_for_model(model_name)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)
