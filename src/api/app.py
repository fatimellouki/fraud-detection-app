"""Flask application factory for the fraud detection API."""

import os
import logging
from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    CORS(app)

    # Load model and scaler at startup
    app.config["MODEL"] = None
    app.config["SCALER"] = None
    app.config["FEATURE_NAMES"] = None

    _load_artifacts(app)

    # Register routes
    from src.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app


def _load_artifacts(app):
    """Load trained model and scaler into app config."""
    import joblib
    from config import MODELS_DIR

    model_path = os.path.join(MODELS_DIR, "stacking_ensemble.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if os.path.exists(model_path):
        app.config["MODEL"] = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. API will return errors.")

    if os.path.exists(scaler_path):
        app.config["SCALER"] = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")


if __name__ == "__main__":
    from config import API_HOST, API_PORT
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=True)
