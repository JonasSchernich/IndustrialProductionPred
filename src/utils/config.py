# src/utils/config.py
from __future__ import annotations
from typing import Any, Dict, Optional
import pathlib
import yaml


def load_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {p}")
    return data


def get_config(
    globals_yaml: Optional[str | pathlib.Path] = None,
    data_yaml: Optional[str | pathlib.Path] = None,
    hardware_yaml: Optional[str | pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Lädt optionale YAML-Dateien und merged sie in ein Dict:
      cfg["globals"], cfg["data"], cfg["hardware"]
    """
    cfg: Dict[str, Any] = {}
    if globals_yaml:
        cfg["globals"] = load_yaml(globals_yaml)
    if data_yaml:
        cfg["data"] = load_yaml(data_yaml)
    if hardware_yaml:
        cfg["hardware"] = load_yaml(hardware_yaml)
    return cfg
