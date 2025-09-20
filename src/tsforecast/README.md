
tsforecast — lightweight time-series forecasting toolbox

Key features:
- Expanding-window rolling forecast (one-step ahead) with progress logging
- Feature engineering: global/per-feature lags, rolling means, EMA, (optional) PCA
- Feature selection on engineered features (variance filter + |Pearson corr|)
- Models: ElasticNet, RandomForest, XGBoost, LightGBM, and baselines (Mean, RandomWalk, AR1)
- Flexible search between FE candidates and HP grid (fast vs. full factorial)

See `tsforecast/rolling/online.py` for orchestration, and `tsforecast/features/*` for FE/FS utilities.
