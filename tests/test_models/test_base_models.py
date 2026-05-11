"""Tests for baseline models."""

import pytest
import numpy as np

from src.models.base_models import create_logistic_regression, create_decision_tree


def test_logistic_regression_creation():
    """Test LR model creation."""
    model = create_logistic_regression()
    assert model is not None
    assert model.max_iter == 1000


def test_logistic_regression_fit_predict(small_train_test):
    """Test LR training and prediction."""
    X_train, X_test, y_train, y_test = small_train_test
    model = create_logistic_regression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_decision_tree_creation():
    """Test DT model creation."""
    model = create_decision_tree()
    assert model is not None
    assert model.max_depth == 10


def test_decision_tree_fit_predict(small_train_test):
    """Test DT training and prediction."""
    X_train, X_test, y_train, y_test = small_train_test
    model = create_decision_tree()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
