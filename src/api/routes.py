"""API endpoints for fraud detection."""

import numpy as np
from flask import Blueprint, request, jsonify, current_app

from config import DEFAULT_THRESHOLD

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_loaded = current_app.config["MODEL"] is not None
    return jsonify({
        "status": "ok" if model_loaded else "degraded",
        "model": "stacking_ensemble_v1",
        "model_loaded": model_loaded,
    })


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Single transaction fraud prediction.

    Request JSON:
        {"features": [float, float, ...]}  (30 feature values)

    Response JSON:
        {"fraud_probability": float, "is_fraud": bool, "threshold": float}
    """
    model = current_app.config["MODEL"]
    scaler = current_app.config["SCALER"]

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    try:
        features = np.array(data["features"], dtype=float).reshape(1, -1)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid features: {str(e)}"}), 400

    # Apply scaling if scaler is available
    if scaler is not None:
        features = scaler.transform(features)

    # Predict
    probability = model.predict_proba(features)[0]
    if len(probability) == 2:
        fraud_prob = float(probability[1])
    else:
        fraud_prob = float(probability[0])

    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    return jsonify({
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": bool(fraud_prob >= threshold),
        "threshold": threshold,
    })


@api_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch transaction fraud prediction.

    Request JSON:
        {"transactions": [[float, ...], [float, ...], ...]}

    Response JSON:
        {"results": [{"fraud_probability": float, "is_fraud": bool}, ...]}
    """
    model = current_app.config["MODEL"]
    scaler = current_app.config["SCALER"]

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "transactions" not in data:
        return jsonify({"error": "Missing 'transactions' in request body"}), 400

    try:
        features = np.array(data["transactions"], dtype=float)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid transactions: {str(e)}"}), 400

    if scaler is not None:
        features = scaler.transform(features)

    probabilities = model.predict_proba(features)
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        fraud_probs = probabilities[:, 1]
    else:
        fraud_probs = probabilities.flatten()

    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    results = [
        {
            "fraud_probability": round(float(p), 4),
            "is_fraud": bool(p >= threshold),
        }
        for p in fraud_probs
    ]

    return jsonify({
        "results": results,
        "total": len(results),
        "flagged": sum(1 for r in results if r["is_fraud"]),
    })


@api_bp.route("/explain", methods=["POST"])
def explain():
    """Prediction with SHAP/LIME explanation.

    Request JSON:
        {"features": [float, float, ...]}

    Response JSON:
        {"fraud_probability": float, "is_fraud": bool,
         "explanation": {"top_contributing_features": [...]}}
    """
    model = current_app.config["MODEL"]
    scaler = current_app.config["SCALER"]

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    try:
        features = np.array(data["features"], dtype=float).reshape(1, -1)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid features: {str(e)}"}), 400

    if scaler is not None:
        features = scaler.transform(features)

    # Prediction
    probability = model.predict_proba(features)[0]
    fraud_prob = float(probability[1]) if len(probability) == 2 else float(probability[0])

    # SHAP explanation (simplified for API speed)
    explanation = {"top_contributing_features": [], "confidence": "unknown"}
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) == 2 else shap_values[0][0]
        else:
            sv = shap_values[0]

        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        if len(sv) > len(feature_names):
            feature_names = [f"Feature_{i}" for i in range(len(sv))]

        # Top contributing features
        indices = np.argsort(np.abs(sv))[::-1][:5]
        top_features = []
        for idx in indices:
            name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            top_features.append({
                "feature": name,
                "contribution": round(float(sv[idx]), 4),
                "direction": "fraud" if sv[idx] > 0 else "legitimate",
            })

        explanation["top_contributing_features"] = top_features
        explanation["confidence"] = "high" if fraud_prob > 0.8 or fraud_prob < 0.2 else "medium"
    except Exception as e:
        explanation["error"] = str(e)

    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    return jsonify({
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": bool(fraud_prob >= threshold),
        "explanation": explanation,
    })


@api_bp.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata."""
    model = current_app.config["MODEL"]

    info = {
        "model": "stacking_ensemble",
        "version": "1.0",
        "model_loaded": model is not None,
        "base_learners": ["RandomForest", "XGBoost", "LightGBM"],
        "meta_learner": "XGBoost",
        "explainability": ["SHAP", "LIME"],
    }

    return jsonify(info)
