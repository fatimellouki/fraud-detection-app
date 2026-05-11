"""Ensemble ML models for fraud detection."""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
    LIGHTGBM_PARAMS, CATBOOST_PARAMS
)


def create_random_forest(**kwargs) -> RandomForestClassifier:
    """Create a Random Forest classifier.

    Args:
        **kwargs: Override default hyperparameters.

    Returns:
        Configured RandomForestClassifier instance.
    """
    params = {**RANDOM_FOREST_PARAMS, **kwargs}
    return RandomForestClassifier(**params)


def create_xgboost(scale_pos_weight: float = None, **kwargs) -> XGBClassifier:
    """Create an XGBoost classifier.

    Args:
        scale_pos_weight: Weight for positive class. If None, not set.
        **kwargs: Override default hyperparameters.

    Returns:
        Configured XGBClassifier instance.
    """
    params = {**XGBOOST_PARAMS, **kwargs}
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def create_lightgbm(**kwargs) -> LGBMClassifier:
    """Create a LightGBM classifier.

    Args:
        **kwargs: Override default hyperparameters.

    Returns:
        Configured LGBMClassifier instance.
    """
    params = {**LIGHTGBM_PARAMS, **kwargs}
    return LGBMClassifier(**params)


def create_catboost(**kwargs) -> CatBoostClassifier:
    """Create a CatBoost classifier.

    Args:
        **kwargs: Override default hyperparameters.

    Returns:
        Configured CatBoostClassifier instance.
    """
    params = {**CATBOOST_PARAMS, **kwargs}
    return CatBoostClassifier(**params)
