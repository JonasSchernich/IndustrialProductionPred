# preset.py
# Minimale, robuste Defaults passend zu deinem Protokoll.

def fs_cfg_default():
    # prewhitened correlation als Default; saison-Lag nur bei Diagnose
    return {
        "mode": "auto_topk_prewhite",
        "topk": 400,                  # Stage A variiert: {200,400,600}
        "tau_scr": None,              # alternativ thresholds: {0.10,0.15,0.20}
        "thresholds": [0.10, 0.15, 0.20],
        "y_lags": (1,),               # D_s = (1, y_{t-1}); saisonal nur wenn Diagnostik dafür spricht
        "seasonal": False,
        "redundancy": {
            "mode": "cluster",        # alternativ "mrmr"
            "tau": 0.90,              # variiere {0.90,0.95}
            "mrmr_k": None            # variiere {80,120,160} falls mode=="mrmr"
        }
    }

def fe_cfg_linear():
    # globale Lags; kein Smoother; externe Blöcke standardmäßig ohne Shift (F_t)
    return {
        "lags_global": [1, 3, 6, 12],
        "rm3": False,
        "external_shift": 0,          # 0 = F_t; 1 = F_{t-1} (bei externer Latenz)
        "tsfresh_path": None,         # bei Nutzung setzen
        "fm_pred_path": None          # Chronos/Found.-Densities (Parquet)
    }

def fe_cfg_tree():
    # per-Feature-Lagwahl; Ranking prewhitened
    return {
        "per_feature": {
            "lag_candidates": [1, 2, 3, 6, 12],
            "topk": 1,                # variiere {1,2}
            "rank": "prewhite"        # "corr" oder "prewhite"
        },
        "rm3": False,
        "external_shift": 0,
        "tsfresh_path": None,
        "fm_pred_path": None
    }

def asha_default():
    # Stage A: 3 Blöcke, je 60 Schritte; effizient und reproduzierbar
    return {
        "fidelities": [1, 2, 3],
        "steps": {1: 60, 2: 60, 3: 60},
        "reduction": 3,
        "grace_iter": 1
    }

# --- Grids (Stage A) ---

def grid_dfm():
    return {
        "r": [2, 3, 4, 5, 6, 8],
        "lag": [0, 1]
    }

def grid_pca_en():
    return {
        "pca_tau": [0.95, 0.99],
        "pca_kmax": [25, 50, 100],
        "alpha": [0.2, 0.4, 0.6, 0.8],
        "lambda_grid": 50  # Log-Grid intern erzeugen
    }

def grid_pls_en():
    return {
        "pls_c": [2, 4, 6, 8],
        "alpha": [0.2, 0.4, 0.6, 0.8],
        "lambda_grid": 50
    }

def grid_lgbm():
    return {
        "max_depth": [2, 3, 4],
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.03, 0.05, 0.10],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.5, 0.8],
        "min_child_samples": [10, 20, 40],
        "reg_lambda": [0, 5, 10, 20],
        "reg_alpha": [0, 1, 5],
        "n_estimators": 1500,   # mit Early Stopping auf Dev-Tail im Train
    }

def grid_xgb():
    return {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.10],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.5, 0.8],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [0, 5, 10, 20],
        "reg_alpha": [0, 1, 5],
        "n_estimators": 1500,
    }

def grid_tabpfn():
    return {
        "dr_k": [32, 64, 96],
        "samples": [8, 16, 32]
    }

# Optional: Feintuning-Nachbarschaften in Stage B
def local_refinements():
    return {
        "pca_kmax": [50, 100, 150],
        "pls_c": [2, 4, 6, 8, 10]
    }
