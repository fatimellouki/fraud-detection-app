"""Tests for class imbalance handling."""

import pytest
import numpy as np

from src.data.balancer import ImbalanceHandler


def test_apply_smote(sample_features_target):
    """Test SMOTE oversampling."""
    X, y = sample_features_target
    handler = ImbalanceHandler()
    X_res, y_res = handler.apply_smote(X, y)

    # Should have more samples
    assert len(y_res) > len(y)
    # Should have more fraud samples
    assert sum(y_res == 1) > sum(y == 1)


def test_apply_adasyn(sample_features_target):
    """Test ADASYN oversampling."""
    X, y = sample_features_target
    handler = ImbalanceHandler()
    X_res, y_res = handler.apply_adasyn(X, y)

    assert len(y_res) > len(y)
    assert sum(y_res == 1) > sum(y == 1)


def test_apply_random_undersampling(sample_features_target):
    """Test random undersampling."""
    X, y = sample_features_target
    handler = ImbalanceHandler()
    X_res, y_res = handler.apply_random_undersampling(X, y)

    # Should have fewer samples
    assert len(y_res) < len(y)
    # Should have fewer legit samples
    assert sum(y_res == 0) < sum(y == 0)


def test_get_class_weights(sample_features_target):
    """Test class weight computation."""
    _, y = sample_features_target
    weights = ImbalanceHandler.get_class_weights(y)

    assert 0 in weights
    assert 1 in weights
    # Fraud weight should be higher (minority class)
    assert weights[1] > weights[0]


def test_compare_techniques(sample_features_target):
    """Test comparison of all imbalance techniques."""
    X, y = sample_features_target
    handler = ImbalanceHandler()
    results = handler.compare_techniques(X, y)

    assert "none" in results
    assert "smote" in results
    assert "cost_sensitive" in results
    assert results["none"]["n_samples"] == len(y)
