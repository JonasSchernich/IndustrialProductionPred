from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import pathlib
import re
import copy
import optuna
import yaml


# ============================================================
# YAML laden
# ============================================================

def load_space_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        space = yaml.safe_load(f) or {}
    if not isinstance(space, dict):
        raise ValueError("search space YAML must be a mapping")
    space.setdefault("feature_space", {})
    space.setdefault("model_space", {})
    space.setdefault("metric", "mae")
    space.setdefault("device", "cpu")
    return space


# ============================================================
# Utilities
# ============================================================

def _is_active(drawn: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    if not cond:
        return True
    return all(drawn.get(k) == v for k, v in cond.items())

def _parse_lags(val) -> tuple[int, ...]:
    """
    Akzeptiert: "1", "1,2,3", [1,2,3], (1,2,3), 1 → (1,2,3)
    """
    if isinstance(val, str):
        parts = re.split(r"[,\s]+", val.strip())
        nums = [int(p) for p in parts if p]
        return tuple(sorted(set(n for n in nums if n > 0)))
    if isinstance(val, (list, tuple)):
        return tuple(sorted(set(int(n) for n in val if int(n) > 0)))
    return (int(val),)


# ============================================================
# PROGRAMMATIC SEARCH SPACE (Fallback, wenn kein YAML benutzt)
# ============================================================

def suggest_feature_params(
    trial: optuna.trial.Trial,
    *,
    pca_groups: Optional[Dict[str, list]] = None,
    groupwise_lags: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    # lags als STRING-Choices, um Optuna-Warnungen (nested types) zu vermeiden
    lags_str = trial.suggest_categorical("lags", ("1", "1,2,3", "1,3,6,12"))
    lag_strategy = trial.suggest_categorical("lag_strategy", ("value", "ema", "mom"))
    ema_span = trial.suggest_int("ema_span", 2, 8) if lag_strategy == "ema" else 3

    use_intercorr = trial.suggest_categorical("use_intercorr", (False, True))
    if use_intercorr:
        intercorr_threshold = trial.suggest_float("intercorr_threshold", 0.85, 0.99)
        intercorr_method = trial.suggest_categorical("intercorr_method", ("var", "first"))
    else:
        intercorr_threshold, intercorr_method = 0.95, "var"

    use_corr_selector = trial.suggest_categorical("use_corr_selector", (False, True))
    if use_corr_selector:
        corr_top_k = trial.suggest_int("corr_top_k", 25, 300, step=25)
        corr_min_abs = trial.suggest_float("corr_min_abs", 0.0, 0.3)
    else:
        corr_top_k, corr_min_abs = 100, 0.0

    use_variance_threshold = trial.suggest_categorical("use_variance_threshold", (False, True))
    var_threshold = trial.suggest_float("var_threshold", 1e-8, 1e-5, log=True) if use_variance_threshold else 0.0

    use_pca = trial.suggest_categorical("use_pca", (False, True))
    if use_pca and pca_groups:
        pca_before_lags = trial.suggest_categorical("pca_before_lags", (False, True))
        pca_n_components = trial.suggest_int("pca_n_components", 1, 5)
    else:
        pca_before_lags, pca_n_components = False, 0

    use_shock_dummy = trial.suggest_categorical("use_shock_dummy", (False, True))
    shock_sigma = trial.suggest_float("shock_sigma", 2.0, 4.0) if use_shock_dummy else None

    return dict(
        manual_features=None,
        include_regex=None,
        exclude_regex=None,
        lags=_parse_lags(lags_str),
        lag_strategy=lag_strategy,
        ema_span=int(ema_span),
        groupwise_lags=groupwise_lags,
        shock_dummy_sigma=float(shock_sigma) if shock_sigma is not None else None,
        use_intercorr=bool(use_intercorr),
        intercorr_threshold=float(intercorr_threshold),
        intercorr_method=str(intercorr_method),
        use_corr_selector=bool(use_corr_selector),
        corr_top_k=int(corr_top_k),
        corr_min_abs=float(corr_min_abs),
        use_variance_threshold=bool(use_variance_threshold),
        var_threshold=float(var_threshold),
        pca_groups=pca_groups if (use_pca and pca_groups) else None,
        pca_n_components=int(pca_n_components) if (use_pca and pca_groups) else 0,
        pca_before_lags=bool(pca_before_lags if (use_pca and pca_groups) else False),
    )


def suggest_model_params(
    trial: optuna.trial.Trial,
    *,
    model: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    m = model.lower()
    if m in ("elasticnet", "enet", "elastic_net"):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return {"model": "elasticnet", "alpha": float(alpha), "l1_ratio": float(l1_ratio), "max_iter": 5000, "random_state": 42}
    raise ValueError(f"Model '{model}' is not supported by programmatic search space yet.")


# ============================================================
# YAML SEARCH SPACE (empfohlen)
# ============================================================

def suggest_feature_params_from_yaml(
    trial: optuna.trial.Trial,
    space: Dict[str, Any],
    *,
    pca_groups: Optional[Dict[str, list]] = None,
    groupwise_lags: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    fs = space.get("feature_space", {})
    drawn: Dict[str, Any] = {}

    def draw(name: str, spec: Dict[str, Any]):
        active_if = spec.get("active_if", {})
        if active_if and not _is_active(drawn, active_if):
            return None

        # Spezialfall 'lags': Choices als Strings (z. B. "1,2,3") gegen Optuna-Warnungen
        if name == "lags" and "choices" in spec:
            choices = []
            for c in spec["choices"]:
                if isinstance(c, (list, tuple)):
                    choices.append(",".join(str(int(v)) for v in c))
                else:
                    choices.append(str(c))
            val = trial.suggest_categorical(name, tuple(choices))
            drawn[name] = val
            return val

        if "choices" in spec:
            val = trial.suggest_categorical(name, tuple(spec["choices"]))
        else:
            typ = spec.get("type", "float")
            low, high = spec["low"], spec["high"]
            log = bool(spec.get("log", False))
            step = spec.get("step")
            if typ == "int":
                val = trial.suggest_int(name, int(low), int(high), step=int(step) if step else 1, log=log)
            else:
                val = trial.suggest_float(name, float(low), float(high), log=log)
        drawn[name] = val
        return val

    for key, spec in fs.items():
        draw(key, spec)

    ema_span = int(drawn.get("ema_span", 3))
    use_pca = bool(drawn.get("use_pca", False))
    lags_tuple = _parse_lags(drawn.get("lags", "1"))

    return dict(
        manual_features=None,
        include_regex=None,
        exclude_regex=None,
        lags=lags_tuple,
        lag_strategy=str(drawn.get("lag_strategy", "value")),
        ema_span=ema_span,
        groupwise_lags=groupwise_lags,
        shock_dummy_sigma=float(drawn["shock_sigma"]) if drawn.get("shock_sigma") is not None else None,
        use_intercorr=bool(drawn.get("use_intercorr", False)),
        intercorr_threshold=float(drawn.get("intercorr_threshold", 0.95)),
        intercorr_method=str(drawn.get("intercorr_method", "var")),
        use_corr_selector=bool(drawn.get("use_corr_selector", False)),
        corr_top_k=int(drawn.get("corr_top_k", 100)),
        corr_min_abs=float(drawn.get("corr_min_abs", 0.0)),
        use_variance_threshold=bool(drawn.get("use_variance_threshold", False)),
        var_threshold=float(drawn.get("var_threshold", 0.0)),
        pca_groups=pca_groups if (use_pca and pca_groups) else None,
        pca_n_components=int(drawn.get("pca_n_components", 0)) if (use_pca and pca_groups) else 0,
        pca_before_lags=bool(drawn.get("pca_before_lags", False)) if (use_pca and pca_groups) else False,
    )


def suggest_model_params_from_yaml(
    trial: optuna.trial.Trial,
    space: Dict[str, Any],
    *,
    model: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    ms = space.get("model_space", {})
    drawn: Dict[str, Any] = {}

    def draw(name: str, spec: Dict[str, Any]):
        if "choices" in spec:
            val = trial.suggest_categorical(name, tuple(spec["choices"]))
        else:
            typ = spec.get("type", "float")
            low, high = spec["low"], spec["high"]
            log = bool(spec.get("log", False))
            step = spec.get("step")
            if typ == "int":
                val = trial.suggest_int(name, int(low), int(high), step=int(step) if step else 1, log=log)
            else:
                val = trial.suggest_float(name, float(low), float(high), log=log)
        drawn[name] = val
        return val

    for key, spec in ms.items():
        draw(key, spec)

    m = model.lower()
    if m in ("elasticnet", "enet", "elastic_net"):
        return {
            "model": "elasticnet",
            "alpha": float(drawn["alpha"]),
            "l1_ratio": float(drawn["l1_ratio"]),
            "max_iter": 5000,
            "random_state": 42,
        }

    raise ValueError(f"Model '{model}' not supported in YAML helper yet.")


# ============================================================
# ESTIMATOR FACTORY
# ============================================================

def build_estimator_fn(model_params: Dict[str, Any], *, device: str = "cpu"):
    """
    Liefert eine 0-Argumente-Factory, die beim Aufruf das Modell-Objekt erzeugt.
    Aktuell: ElasticNet (Wrapper aus src.modeling.linear).
    """
    if "model" not in model_params:
        raise ValueError("model_params must contain a 'model' key")
    mp = copy.deepcopy(model_params)
    model_name = str(mp.pop("model")).lower()

    if model_name in ("elasticnet", "enet", "elastic_net"):
        # robust gegen beide Klassennamen
        try:
            from src.modeling.linear import ElasticNetRegressor as _EN
        except ImportError:
            from src.modeling.linear import ElasticNetWrapper as _EN

        def factory():
            return _EN(**mp)
        return factory

    # Platzhalter für spätere Erweiterungen …
    raise ValueError(f"build_estimator_fn: unknown model '{model_name}'")
