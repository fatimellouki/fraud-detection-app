"""Request/response JSON schemas for API validation."""

PREDICT_REQUEST = {
    "type": "object",
    "required": ["features"],
    "properties": {
        "features": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Array of 30 feature values (V1-V28, Amount, Time)",
        },
        "threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.5,
            "description": "Classification threshold",
        },
    },
}

PREDICT_RESPONSE = {
    "type": "object",
    "properties": {
        "fraud_probability": {"type": "number"},
        "is_fraud": {"type": "boolean"},
        "threshold": {"type": "number"},
    },
}

BATCH_REQUEST = {
    "type": "object",
    "required": ["transactions"],
    "properties": {
        "transactions": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
            "description": "Array of transaction feature arrays",
        },
        "threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.5,
        },
    },
}

EXPLAIN_RESPONSE = {
    "type": "object",
    "properties": {
        "fraud_probability": {"type": "number"},
        "is_fraud": {"type": "boolean"},
        "explanation": {
            "type": "object",
            "properties": {
                "top_contributing_features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string"},
                            "contribution": {"type": "number"},
                            "direction": {"type": "string"},
                        },
                    },
                },
                "confidence": {"type": "string"},
            },
        },
    },
}
