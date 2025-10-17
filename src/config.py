# src/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Literal, TypedDict, Optional, Dict, Any, Tuple
from pathlib import Path
import os

# ---------- Projekt-Root robust auflösen ----------
def _resolve_root() -> Path:
    # 1) Umgebungsvariable bevorzugen (z. B. im Notebook gesetzt)
    env = os.environ.get("PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    # 2) Fallback: Elternordner von src/
    return Path(__file__).resolve().parent.parent

ROOT = _resolve_root()

# ---------- Ordnerstruktur ----------
PROCESSED = ROOT / "data" / "processed"
OUTPUTS   = ROOT / "outputs"
LOGS      = OUTPUTS / "logs"
STAGEA_DIR = OUTPUTS / "stageA"
STAGEB_DIR = OUTPUTS / "stageB"

for _p in [PROCESSED, OUTPUTS, LOGS, STAGEA_DIR, STAGEB_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ---------- CorrelationSpec ----------
CorrelationMode = Literal["expanding", "ewma"]

class CorrelationSpec(TypedDict):
    mode: CorrelationMode
    window: Optional[int]
    lam: Optional[float]

DEFAULT_CORR_SPEC: CorrelationSpec = {"mode": "expanding", "window": None, "lam": None}
EWMA_CORR_SPEC:    CorrelationSpec = {"mode": "ewma", "window": 60, "lam": 0.98}


# ---------- Stage-Defaults ----------
DEFAULT_W0_A = 180
DEFAULT_BLOCKS_A: Tuple[Tuple[int,int], ...] = ((181,200), (201,220), (221,240))
DEFAULT_W0_B = 240

# ---------- Globale Konfiguration ----------
@dataclass
class GlobalConfig:
    # Seeds & Refresh
    seed: int = 123
    refresh_cadence_months: int = 12

    # Nuisance & Korrelation
    corr_spec: CorrelationSpec = field(default_factory=lambda: dict(DEFAULT_CORR_SPEC))

    # Feature Engineering
    lag_candidates: tuple = tuple(range(1, 11))
    top_k_lags_per_feature: int = 1
    use_rm3: bool = True

    # Screening
    k1_topk: int = 50
    screen_threshold: Optional[float] = None

    # Redundanz
    redundancy_method: Literal["cluster", "greedy"] = "greedy"
    redundancy_param: float = 0.9  # Schwelle (greedy) bzw. tau (cluster)

    # Dimension Reduction
    dr_method: Literal["none", "pca", "pls"] = "none"
    pca_var_target: float = 0.95
    pca_kmax: int = 25
    pls_components: int = 2

    # Stage A/B Splits
    W0_A: int = DEFAULT_W0_A
    BLOCKS_A: Tuple[Tuple[int,int], ...] = DEFAULT_BLOCKS_A
    W0_B: int = DEFAULT_W0_B

    # Online-Policy
    policy_window: int = 12
    policy_gain_min: float = 0.03
    policy_cooldown: int = 3

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["corr_spec"] = dict(self.corr_spec)
        return d

def outputs_for_model(model_name: str) -> Dict[str, Path]:
    """Erstellt/liefert Modell-spezifische Output-Ordner."""
    mdirA = STAGEA_DIR / model_name
    mdirB = STAGEB_DIR / model_name
    for sub in ["block1", "block2", "block3"]:
        (mdirA / sub).mkdir(parents=True, exist_ok=True)
    (mdirB / "monthly").mkdir(parents=True, exist_ok=True)
    (mdirB / "summary").mkdir(parents=True, exist_ok=True)
    return {"stageA": mdirA, "stageB": mdirB, "logs": LOGS / f"{model_name}.log"}
