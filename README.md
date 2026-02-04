 # Macroeconomic Nowcasting of German Industrial Production with High-Dimensional Sentiment Data

Code repository for the master thesis **“A Systematic Comparison of Machine Learning Methods for Macroeconomic Nowcasting with High-Dimensional Sentiment Data: Evidence from German Industrial Production”**.

**Goal:** Nowcast the monthly change in German industrial production using high-dimensional ifo Business Survey  indicators ($p \gg T$), optionally augmented with target-history feature blocks.
The following three research questions are adressed:

1. Which ML and DL methods deliver the most accurate nowcasts of monthly changes in IP?
2. How do these methods perform relative to simple benchmarks commonly used in the now-
casting literature?
Economic:
3. Which ifo Business Survey-based features drive the predictions and are these drivers stable
over time?
---

## 📂 Repository Structure

- `data/`
  - `raw/` – Raw input data.
  - `processed/` – Generated, model-ready datasets (CSV/Parquet).
- `notebooks/`
  - `descriptive_analysis/` – Exploratory analysis (`analysis.ipynb` + figures).
  - `feature_importance/` – Dynamic feature selection pipeline (`feature_importance.ipynb`).
  - `target_based/` – Feature generation based on target history (`AR.ipynb`, `chronos.ipynb`, `tsfresh_slim.ipynb`).
  - `modeling/` – Core modeling notebooks (EN, SVR, GPR, ET, LGBM, TabPFN, SFM, Ensemble).
  - `tests/` – Final evaluation and result visualization (`evaluations.ipynb` + figures).
  - `baselines.ipynb` – Benchmark models (Random Walk, AR, etc.).
- `src/` – Core Python package (data loading, feature engineering, tuning, model wrappers).
- `outputs/` – Experiment artifacts (Stage A/B results, feature importance logs, figures).

---

## 🚀 Quickstart

### 1. Place Raw Data
Raw data is **not** included in the repository. Place the required files in `data/raw/` exactly as follows:
- `data/raw/features/bdi1a_*.xlsx` (ifo survey features)
- `data/raw/target/IndustrialProd.xlsx` (Target file; sheet `IP`)

### 2. Build Processed Datasets
Run the data loading script from the repository root:

```bash
python -m src.data.load_data
```

*This generates `target.csv`, `features.csv`, and cleaned datasets in `data/processed/`.*

---
-> All files are already included/pregenerated in this repository
## 🛠️ Typical Workflow

### A) Descriptive Analysis (Optional)
Run `notebooks/descriptive_analysis/analysis.ipynb` to generate initial plots and sanity checks.

### B) Generate Target-Based Features (Setup II)
Run any of the notebooks in `notebooks/target_based/` to create additional feature blocks (e.g., `AR.parquet`, `chronos_*.parquet`) in `data/processed/`. These can be appended to ifo features during modeling.

### C) Dynamic Feature Importance (Setup III)
Run `notebooks/feature_importance/feature_importance.ipynb`.
- **Output:** Generates `X_eng_full_lagged.parquet` and rolling importance artifacts in `outputs/feature_importance/`.
- **Note:** Includes automatic GPU/CPU fallback for LightGBM.

### D) Baselines
Run `notebooks/baselines.ipynb` to establish benchmark metrics (e.g., AR baselines).

### E) Modeling (Tuning + Evaluation)
Execute notebooks in `notebooks/modeling/` (e.g., `EN.ipynb`, `LGBM.ipynb`, `ensemble.ipynb`).
- **Stage A:** Block holdout tuning to prune configurations.
- **Stage B:** Monthly rolling evaluation with online model selection.
- **Dynamic FI Switch:** Set `USE_DYNAMIC_FI_PIPELINE = True` in notebooks to utilize artifacts from step (C).

### F) Results & Visualization
Run `notebooks/tests/evaluations.ipynb` to aggregate results from `outputs/` and generate final comparison figures.

---

## 🐍 Core Modules (`src/`)

| Module | Description |
| :--- | :--- |
| `src.data.load_data` | ETL script: transforms raw Excel files into processed CSV/Parquet matrices. |
| `src.config` | Global settings, "thesis" presets, correlation specs, and directory helpers. |
| `src.io_timesplits` | Data loaders (target, ifo features, artifacts) and time-split utilities. |
| `src.features` | Feature engineering: lag expansion, screening, redundancy reduction, PCA/PLS. |
| `src.tuning` | Orchestration for Stage A (pruning) and Stage B (rolling evaluation). |
| `src.models` | Unified wrappers for EN, SVR, GPR, ET, LGBM, TabPFN, SFM, and Ensembles. |

---

## ⚙️ Reproducibility & Defaults
- **Seeds:** Controlled via `GlobalConfig` and notebook-level constants.
- **Time-Splits:** Defined in `src/config.py` (default preset: `"thesis"`).
- **Selection Policy:** Stage B uses a decayed rolling window controlled by `policy_window` and `selection_mode`.

### Minimal Run Order
1. `python -m src.data.load_data`
2. `notebooks/baselines.ipynb`
3. `notebooks/modeling/*.ipynb` (Run desired models)
4. `notebooks/tests/evaluations.ipynb`
