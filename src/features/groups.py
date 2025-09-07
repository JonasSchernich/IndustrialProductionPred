# src/features/groups.py
from __future__ import annotations
from typing import Dict, Iterable, List
import pandas as pd


def groups_from_columns(
    columns: Iterable[str],
    split_on: str = ".",
) -> Dict[str, List[str]]:
    """
    Leitet Gruppen automatisch aus Spaltennamen ab, die wie 'Branche.Indicator' strukturiert sind.
    Gibt ein Dict[group_name -> [prefix]] zurück, das du in PCAByGroup/PerGroupTransformer verwenden kannst.
    Beispiel:
      'Auto.Orders', 'Auto.Sales', 'Chemie.Index' → {'Auto': ['Auto.'], 'Chemie': ['Chemie.']}
    """
    groups = {}
    for c in columns:
        if split_on in c:
            g = c.split(split_on, 1)[0]
        else:
            g = "ALL"
        prefix = f"{g}{split_on}"
        groups.setdefault(g, [])
        if prefix not in groups[g]:
            groups[g].append(prefix)
    return groups


def infer_groups_from_dataframe(df: pd.DataFrame, split_on: str = ".") -> Dict[str, List[str]]:
    """Bequemlichkeit: ruft groups_from_columns(df.columns)."""
    return groups_from_columns(df.columns, split_on=split_on)
