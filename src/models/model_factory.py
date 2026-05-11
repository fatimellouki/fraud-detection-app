"""Factory for creating model instances by name."""

from src.models.base_models import create_logistic_regression, create_decision_tree
from src.models.ensemble_models import (
    create_random_forest, create_xgboost, create_lightgbm, create_catboost
)
from src.models.stacking_model import FraudStackingEnsemble


MODEL_REGISTRY = {
    "logistic_regression": create_logistic_regression,
    "decision_tree": create_decision_tree,
    "random_forest": create_random_forest,
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
    "catboost": create_catboost,
    "stacking": lambda **kw: FraudStackingEnsemble(**kw),
}


def create_model(name: str, **kwargs):
    """Create a model instance by name.

    Args:
        name: One of 'logistic_regression', 'decision_tree', 'random_forest',
              'xgboost', 'lightgbm', 'catboost', 'mlp', 'autoencoder', 'stacking'.
        **kwargs: Model-specific hyperparameters.

    Returns:
        Model instance.

    Raises:
        ValueError: If name is not recognized.
    """
    if name == "mlp":
        from src.models.deep_models import MLPFraudClassifier
        input_dim = kwargs.pop("input_dim", 30)
        return MLPFraudClassifier(input_dim=input_dim, **kwargs)

    if name == "autoencoder":
        from src.models.deep_models import AutoencoderDetector
        input_dim = kwargs.pop("input_dim", 30)
        return AutoencoderDetector(input_dim=input_dim, **kwargs)

    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + ["mlp", "autoencoder"]
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )

    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys()) + ["mlp", "autoencoder"]
