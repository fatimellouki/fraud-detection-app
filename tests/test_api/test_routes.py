"""Tests for Flask API endpoints."""

import pytest
import json
import numpy as np
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.app import create_app


@pytest.fixture
def app():
    """Create test Flask app with mocked model."""
    app = create_app()

    # Mock the model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
    mock_model.predict.return_value = np.array([1])
    app.config["MODEL"] = mock_model
    app.config["SCALER"] = None

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def test_health_endpoint(client):
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    """Test single prediction."""
    response = client.post("/predict",
        json={"features": [0.0] * 30},
        content_type="application/json"
    )
    assert response.status_code == 200

    data = response.get_json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "threshold" in data
    assert 0 <= data["fraud_probability"] <= 1


def test_predict_missing_features(client):
    """Test prediction with missing features."""
    response = client.post("/predict",
        json={},
        content_type="application/json"
    )
    assert response.status_code == 400


def test_batch_predict_endpoint(client, app):
    """Test batch prediction."""
    # Update mock for batch
    app.config["MODEL"].predict_proba.return_value = np.array([
        [0.15, 0.85],
        [0.90, 0.10],
    ])

    response = client.post("/predict/batch",
        json={"transactions": [[0.0] * 30, [1.0] * 30]},
        content_type="application/json"
    )
    assert response.status_code == 200

    data = response.get_json()
    assert "results" in data
    assert data["total"] == 2


def test_model_info_endpoint(client):
    """Test model info."""
    response = client.get("/model/info")
    assert response.status_code == 200

    data = response.get_json()
    assert data["model"] == "stacking_ensemble"
    assert "SHAP" in data["explainability"]
