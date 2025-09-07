# src/utils/io.py
from __future__ import annotations
from typing import Optional, Iterable
import pathlib
import pandas as pd


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_panel(path: str | pathlib.Path, parse_dates: bool = True) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True if parse_dates else False)


def write_df(df: pd.DataFrame, path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p)
            return
        except Exception:
            # fallback auf CSV
            p = p.with_suffix(".csv")
    df.to_csv(p)


def list_files(pattern: str | pathlib.Path) -> list[pathlib.Path]:
    p = pathlib.Path(pattern)
    if p.is_dir():
        return sorted(p.iterdir())
    return sorted(pathlib.Path().glob(str(pattern)))


def load_features_target(features_path: str | pathlib.Path, target_path: str | pathlib.Path):
    X = read_panel(features_path, parse_dates=True)
    y = pd.read_csv(target_path, index_col=0, parse_dates=True).iloc[:, 0]
    common_index = X.index.intersection(y.index)
    return X.loc[common_index], y.loc[common_index]
