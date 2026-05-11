"""Model serialization utilities."""

import os
import json
import logging
from datetime import datetime
import joblib

from config import MODELS_DIR

logger = logging.getLogger(__name__)


def save_model(model, name: str, metrics: dict = None):
    """Save a trained model and update metadata.

    Args:
        model: Trained model instance.
        name: Model name (used for filename).
        metrics: Optional metrics dictionary to store in metadata.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")

    if metrics:
        _update_metadata(name, metrics)


def load_model(name: str):
    """Load a saved model.

    Args:
        name: Model name (filename without extension).

    Returns:
        Loaded model instance.
    """
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model


def _update_metadata(model_name: str, metrics: dict):
    """Update the models metadata JSON file.

    Args:
        model_name: Name of the model.
        metrics: Metrics dictionary.
    """
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Filter out non-serializable values
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, list)):
            clean_metrics[k] = v

    metadata[model_name] = {
        "metrics": clean_metrics,
        "saved_at": datetime.now().isoformat(),
        "version": "1.0",
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
