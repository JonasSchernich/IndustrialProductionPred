from __future__ import annotations
from typing import Optional, Dict, List, Tuple

from sklearn.pipeline import Pipeline

from .transforms import (
    IdentityTransformer,
    ColumnSelector,
    LagMaker,
    ShockMonthDummyFromTarget,
    PerGroupTransformer,
    FeatureFrameUnion,
)
from .selection import (
    InterFeatureCorrSelector,
    CorrWithTargetSelector,
    VarianceThresholdColumns,
    PCAByGroup,
)


def _core_steps(
    *,
    manual_features: Optional[List[str]],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    lags: Tuple[int, ...],
    lag_strategy: str,
    ema_span: int,
    groupwise_lags: Optional[Dict[str, List[str]]],
    use_variance_threshold: bool,
    var_threshold: float,
    use_corr_selector: bool,
    corr_top_k: int,
    corr_min_abs: float,
    use_intercorr: bool,
    intercorr_threshold: float,
    intercorr_method: str,
    use_pca: bool,
    pca_groups: Optional[Dict[str, List[str]]],
    pca_n_components: int,
    pca_before_lags: bool,
) -> Pipeline:
    steps = []

    # 1) Spaltenauswahl
    steps.append(("select", ColumnSelector(
        manual_features=manual_features,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )))

    # 2) (optional) PCA vor Lags
    if use_pca and pca_groups and pca_n_components > 0 and pca_before_lags:
        steps.append(("pca", PCAByGroup(groups=pca_groups, n_components=int(pca_n_components))))

    # 3) Lags (ggf. gruppenweise)
    lagger = LagMaker(lags=lags, strategy=lag_strategy, ema_span=int(ema_span))
    if groupwise_lags:
        steps.append(("lags", PerGroupTransformer(base_transformer=lagger, groups=groupwise_lags)))
    else:
        steps.append(("lags", lagger))

    # 4) (optional) PCA nach Lags
    if use_pca and pca_groups and pca_n_components > 0 and not pca_before_lags:
        steps.append(("pca_after", PCAByGroup(groups=pca_groups, n_components=int(pca_n_components))))

    # 5) (optional) Feature-Filter
    if use_variance_threshold:
        steps.append(("var_th", VarianceThresholdColumns(threshold=float(var_threshold))))
    if use_intercorr:
        steps.append(("intercorr", InterFeatureCorrSelector(threshold=float(intercorr_threshold),
                                                            method=str(intercorr_method))))
    if use_corr_selector:
        steps.append(("corr_to_target", CorrWithTargetSelector(top_k=int(corr_top_k),
                                                               min_abs=float(corr_min_abs))))

    if len(steps) == 0:
        steps = [("identity", IdentityTransformer())]

    return Pipeline(steps)


def make_feature_pipeline(
    *,
    manual_features: Optional[List[str]] = None,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    lags: Tuple[int, ...] = (1,),
    lag_strategy: str = "value",
    ema_span: int = 3,
    groupwise_lags: Optional[Dict[str, List[str]]] = None,
    shock_dummy_sigma: Optional[float] = None,
    use_intercorr: bool = False,
    intercorr_threshold: float = 0.95,
    intercorr_method: str = "var",
    use_corr_selector: bool = False,
    corr_top_k: int = 100,
    corr_min_abs: float = 0.0,
    use_variance_threshold: bool = False,
    var_threshold: float = 0.0,
    pca_groups: Optional[Dict[str, List[str]]] = None,
    pca_n_components: int = 0,
    pca_before_lags: bool = False,
) -> Pipeline:
    """
    Baut die komplette Feature-Pipeline. Bei shock_dummy_sigma wird die Kernpipeline
    mit einem Shock-Dummy per Union erweitert.
    """
    core = _core_steps(
        manual_features=manual_features,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        lags=tuple(lags),
        lag_strategy=lag_strategy,
        ema_span=int(ema_span),
        groupwise_lags=groupwise_lags,
        use_variance_threshold=use_variance_threshold,
        var_threshold=float(var_threshold),
        use_corr_selector=use_corr_selector,
        corr_top_k=int(corr_top_k),
        corr_min_abs=float(corr_min_abs),
        use_intercorr=use_intercorr,
        intercorr_threshold=float(intercorr_threshold),
        intercorr_method=str(intercorr_method),
        use_pca=bool(pca_groups and pca_n_components > 0),
        pca_groups=pca_groups,
        pca_n_components=int(pca_n_components),
        pca_before_lags=bool(pca_before_lags),
    )

    if shock_dummy_sigma is None:
        return core

    # Kernfeatures + Shock-Dummy zusammenführen
    union = FeatureFrameUnion([
        ("core", core),
        ("shock", ShockMonthDummyFromTarget(sigma=float(shock_dummy_sigma))),
    ])
    return union
