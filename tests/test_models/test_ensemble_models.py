"""Tests for ensemble models."""

import pytest
import numpy as np

from src.models.ensemble_models import (
    create_random_forest, create_xgboost, create_lightgbm
)


def test_random_forest_fit_predict(small_train_test):
    """Test Random Forest training and prediction."""
    X_train, X_test, y_train, y_test = small_train_test
    model = create_random_forest(n_estimators=10)  # Small for speed
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(preds) == len(y_test)
    assert proba.shape[1] == 2


def test_xgboost_fit_predict(small_train_test):
    """Test XGBoost training and prediction."""
    X_train, X_test, y_train, y_test = small_train_test
    model = create_xgboost(n_estimators=10)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(preds) == len(y_test)
    assert proba.shape[1] == 2


def test_lightgbm_fit_predict(small_train_test):
    """Test LightGBM training and prediction."""
    X_train, X_test, y_train, y_test = small_train_test
    model = create_lightgbm(n_estimators=10)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(preds) == len(y_test)
    assert proba.shape[1] == 2


def test_xgboost_with_scale_pos_weight(small_train_test):
    """Test XGBoost with scale_pos_weight."""
    X_train, X_test, y_train, y_test = small_train_test
    model = create_xgboost(scale_pos_weight=50, n_estimators=10)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    assert proba.shape[1] == 2
