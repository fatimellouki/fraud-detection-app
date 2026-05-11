"""Evaluation metrics for fraud detection models."""

import time
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def compute_all_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute all evaluation metrics in one call.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions.
        y_proba: Fraud probability scores.

    Returns:
        Dictionary with all metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc_roc": compute_auc_roc(y_true, y_proba),
        "auprc": compute_auprc(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_auc_roc(y_true, y_proba) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels.
        y_proba: Fraud probability scores.

    Returns:
        AUC-ROC score.
    """
    try:
        return round(roc_auc_score(y_true, y_proba), 4)
    except ValueError:
        return 0.0


def compute_auprc(y_true, y_proba) -> float:
    """Compute Area Under the Precision-Recall Curve.

    Args:
        y_true: Ground truth labels.
        y_proba: Fraud probability scores.

    Returns:
        AUPRC score.
    """
    try:
        return round(average_precision_score(y_true, y_proba), 4)
    except ValueError:
        return 0.0


def compute_f1(y_true, y_pred) -> float:
    """Compute F1-score."""
    return round(f1_score(y_true, y_pred, zero_division=0), 4)


def compute_precision_recall(y_true, y_pred) -> tuple:
    """Compute precision and recall.

    Returns:
        Tuple of (precision, recall).
    """
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return round(p, 4), round(r, 4)


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Compute 2x2 confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(y_true, y_pred) -> str:
    """Generate full classification report."""
    return classification_report(
        y_true, y_pred,
        target_names=["Légitime", "Fraude"],
        zero_division=0
    )


def compute_inference_time(model, X_sample, n_runs: int = 10) -> float:
    """Measure average inference time per sample in milliseconds.

    Args:
        model: Trained model with predict or predict_proba method.
        X_sample: Sample feature matrix.
        n_runs: Number of runs to average.

    Returns:
        Average milliseconds per prediction.
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        if hasattr(model, "predict_proba"):
            model.predict_proba(X_sample)
        else:
            model.predict(X_sample)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    n_samples = len(X_sample) if hasattr(X_sample, "__len__") else 1
    return round((avg_time / n_samples) * 1000, 4)
