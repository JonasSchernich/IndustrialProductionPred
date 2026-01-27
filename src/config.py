# src/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Literal, TypedDict, Optional, Dict, Any, Tuple
from pathlib import Path
import os


# ---------- Projekt-Root ----------
def _resolve_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(__file__).resolve().parent.parent


ROOT = _resolve_root()

# ---------- Folder Structure ----------
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
LOGS = OUTPUTS / "logs"
STAGEA_DIR = OUTPUTS / "stageA"
STAGEB_DIR = OUTPUTS / "stageB"

for _p in [PROCESSED, OUTPUTS, LOGS, STAGEA_DIR, STAGEB_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ---------- CorrelationSpec ----------
CorrelationMode = Literal["expanding", "ewma"]


class CorrelationSpec(TypedDict):
    mode: CorrelationMode
    lam: Optional[float]


DEFAULT_CORR_SPEC: CorrelationSpec = {"mode": "expanding", "lam": None}
EWMA_CORR_SPEC: CorrelationSpec = {"mode": "ewma", "lam": 0.98}


# ---------- Stage-Defaults ----------
DEFAULT_W0_A: int = 180
DEFAULT_BLOCKS_A: Tuple[Tuple[int, int], ...] = ((181, 200), (201, 220), (221, 240))
DEFAULT_W0_B: int = 240

THESIS_SPLITS = dict(W0_A=180, BLOCKS_A=((181, 200), (201, 220), (221, 240)), W0_B=240)
FAST_DEBUG = dict(W0_A=48, BLOCKS_A=((49, 60), (61, 72)), W0_B=73)


# ---------- Globale Configs ----------
@dataclass
class GlobalConfig:
    # Seeds & Refresh
    seed: int = 123
    refresh_cadence_months: int = 12

    # Nuisance & Correlation
    corr_spec: CorrelationSpec = field(default_factory=lambda: dict(DEFAULT_CORR_SPEC))

    # Feature Engineering
    lag_candidates: tuple = tuple(range(0, 7))
    # top_k_lags_per_feature entfernt/ignoriert, da wir jetzt fixe Lags nehmen

    # Screening
    k1_topk: int = 50
    screen_threshold: Optional[float] = None

    # Redundance
    redundancy_method: Literal["greedy"] = "greedy"
    redundancy_param: float = 0.9

    # Dimension Reduction
    dr_method: Literal["none", "pca", "pls"] = "none"
    pca_var_target: float = 0.95
    pca_kmax: int = 25
    pls_components: int = 2

    # Preset Selection
    preset: str = "thesis"

    # Stage A/B Splits
    W0_A: int = DEFAULT_W0_A
    BLOCKS_A: Tuple[Tuple[int, int], ...] = DEFAULT_BLOCKS_A
    W0_B: int = DEFAULT_W0_B

    # Online-Policy
    policy_window: int = 24
    policy_decay: float = 0.97
    selection_mode: str = "decayed_best"

    def __post_init__(self):
        if self.preset == "thesis":
            splits = THESIS_SPLITS
        else:
            splits = FAST_DEBUG

        self.W0_A = splits["W0_A"]
        self.BLOCKS_A = splits["BLOCKS_A"]
        self.W0_B = splits["W0_B"]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["corr_spec"] = dict(self.corr_spec)
        return d


def cfg() -> GlobalConfig:
    return GlobalConfig(preset="thesis")


def outputs_for_model(model_name: str) -> Dict[str, Path]:
    mdirA = STAGEA_DIR / model_name
    mdirB = STAGEB_DIR / model_name
    for sub in ["block1", "block2", "block3"]:
        (mdirA / sub).mkdir(parents=True, exist_ok=True)
    (mdirB / "monthly").mkdir(parents=True, exist_ok=True)
    (mdirB / "summary").mkdir(parents=True, exist_ok=True)
    return {
        "stageA": mdirA,
        "stageB": mdirB,
        "logs": LOGS / f"{model_name}.log",
        "monthly_preds": mdirB / "monthly" / "preds.csv",
        "monthly_scores": mdirB / "monthly" / "scores.csv",
    }