from __future__ import annotations
from typing import Dict, Any

# Falls bei dir schon vorhanden: build_estimator import behalten
from .xgboost import build_estimator as build_xgb  # optional guard in deiner Struktur
try:
    from .lightgbm import build_estimator as build_lgbm
except Exception:
    build_lgbm = None
from .baselines import build_estimator as build_baselines  # falls vorhanden
# ... weitere Modelle importieren wie in deinem Projekt üblich

# Einheitlicher Entry-Point
def build_estimator(model_name: str, params: Dict[str, Any]):
    name = model_name.lower()
    if name in ("xgb", "xgboost"):
        return build_xgb(params)
    if name in ("lgbm", "lightgbm"):
        if build_lgbm is None:
            raise ImportError("LightGBM not available")
        return build_lgbm(params)
    # Baselines (rw, mean, ar1 etc.)
    return build_baselines(model_name, params)

# Fähigkeiten (minimal)
MODEL_CAPS: Dict[str, Dict[str, bool]] = {
    "xgb": {"supports_es": True},
    "xgboost": {"supports_es": True},
    "lgbm": {"supports_es": True},
    "lightgbm": {"supports_es": True},
    "mean": {"supports_es": False},
    "avg": {"supports_es": False},
    "average": {"supports_es": False},
    "randomwalk": {"supports_es": False},
    "rw": {"supports_es": False},
    "naive": {"supports_es": False},
    "ar1": {"supports_es": False},
    "ar(1)": {"supports_es": False},
    # ergänze weitere Modelle bei Bedarf
}

def supports_es(model_name: str) -> bool:
    return MODEL_CAPS.get(model_name.lower(), {}).get("supports_es", False)
