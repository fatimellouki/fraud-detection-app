"""Tests for stacking ensemble model."""

import pytest
import numpy as np

from src.models.stacking_model import FraudStackingEnsemble


def test_stacking_creation():
    """Test stacking ensemble creation."""
    model = FraudStackingEnsemble(cv_folds=2)
    assert model.model is not None


def test_stacking_fit_predict(small_train_test):
    """Test stacking ensemble training and prediction."""
    X_train, X_test, y_train, y_test = small_train_test

    # Use small models for test speed
    model = FraudStackingEnsemble(
        rf_params={"n_estimators": 5},
        xgb_params={"n_estimators": 5},
        lgbm_params={"n_estimators": 5},
        meta_params={"n_estimators": 5, "max_depth": 2, "random_state": 42},
        cv_folds=2
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(preds) == len(y_test)
    assert proba.shape == (len(y_test), 2)
    assert set(preds).issubset({0, 1})


def test_stacking_base_predictions(small_train_test):
    """Test retrieving base model predictions."""
    X_train, X_test, y_train, y_test = small_train_test

    model = FraudStackingEnsemble(
        rf_params={"n_estimators": 5},
        xgb_params={"n_estimators": 5},
        lgbm_params={"n_estimators": 5},
        meta_params={"n_estimators": 5, "max_depth": 2, "random_state": 42},
        cv_folds=2
    )
    model.fit(X_train, y_train)

    base_preds = model.get_base_predictions(X_test)
    assert "rf" in base_preds
    assert "xgb" in base_preds
    assert "lgbm" in base_preds
    assert len(base_preds["rf"]) == len(y_test)
