"""Tests for evaluation metrics."""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_all_metrics, compute_auc_roc, compute_auprc,
    compute_f1, compute_precision_recall, compute_confusion_matrix,
    compute_classification_report
)


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.9, 0.85, 0.4, 0.05, 0.1])
    return y_true, y_pred, y_proba


def test_compute_all_metrics(sample_predictions):
    """Test that compute_all_metrics returns all expected keys."""
    y_true, y_pred, y_proba = sample_predictions
    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    assert "auc_roc" in metrics
    assert "auprc" in metrics
    assert "f1_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "accuracy" in metrics
    assert "specificity" in metrics
    assert "confusion_matrix" in metrics
    assert "tp" in metrics
    assert "fp" in metrics
    assert "tn" in metrics
    assert "fn" in metrics


def test_auc_roc_range(sample_predictions):
    """Test AUC-ROC is between 0 and 1."""
    y_true, _, y_proba = sample_predictions
    auc = compute_auc_roc(y_true, y_proba)
    assert 0 <= auc <= 1


def test_auprc_range(sample_predictions):
    """Test AUPRC is between 0 and 1."""
    y_true, _, y_proba = sample_predictions
    auprc = compute_auprc(y_true, y_proba)
    assert 0 <= auprc <= 1


def test_f1_range(sample_predictions):
    """Test F1 is between 0 and 1."""
    y_true, y_pred, _ = sample_predictions
    f1 = compute_f1(y_true, y_pred)
    assert 0 <= f1 <= 1


def test_precision_recall_range(sample_predictions):
    """Test precision and recall are between 0 and 1."""
    y_true, y_pred, _ = sample_predictions
    p, r = compute_precision_recall(y_true, y_pred)
    assert 0 <= p <= 1
    assert 0 <= r <= 1


def test_confusion_matrix_shape(sample_predictions):
    """Test confusion matrix is 2x2."""
    y_true, y_pred, _ = sample_predictions
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_confusion_matrix_sums(sample_predictions):
    """Test confusion matrix sums equal total samples."""
    y_true, y_pred, _ = sample_predictions
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.sum() == len(y_true)


def test_classification_report_type(sample_predictions):
    """Test classification report returns string."""
    y_true, y_pred, _ = sample_predictions
    report = compute_classification_report(y_true, y_pred)
    assert isinstance(report, str)
    assert "Fraude" in report
