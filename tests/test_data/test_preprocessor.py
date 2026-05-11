"""Tests for preprocessing pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessor import FraudPreprocessor


def test_init_standard_scaler():
    """Test initialization with StandardScaler."""
    pp = FraudPreprocessor(scaler_type="standard")
    assert pp.scaler_type == "standard"
    assert pp._fitted is False


def test_init_minmax_scaler():
    """Test initialization with MinMaxScaler."""
    pp = FraudPreprocessor(scaler_type="minmax")
    assert pp.scaler_type == "minmax"


def test_init_invalid_scaler():
    """Test initialization with invalid scaler type."""
    with pytest.raises(ValueError):
        FraudPreprocessor(scaler_type="invalid")


def test_handle_missing_no_nulls(sample_kaggle_data):
    """Test that clean data passes through unchanged."""
    pp = FraudPreprocessor()
    df_clean = pp.handle_missing(sample_kaggle_data)
    assert df_clean.isnull().sum().sum() == 0


def test_handle_missing_with_nulls():
    """Test that nulls are filled."""
    pp = FraudPreprocessor()
    df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, 7, 8]})
    df_clean = pp.handle_missing(df)
    assert df_clean.isnull().sum().sum() == 0


def test_engineer_features_kaggle(sample_kaggle_data):
    """Test Kaggle CC feature engineering."""
    pp = FraudPreprocessor()
    df_eng = pp.engineer_features_kaggle(sample_kaggle_data)

    assert "Hour" in df_eng.columns
    assert "Amount_Log" in df_eng.columns
    assert "V_Mean" in df_eng.columns
    assert "V_Std" in df_eng.columns

    # Hour should be between 0 and 23
    assert df_eng["Hour"].min() >= 0
    assert df_eng["Hour"].max() <= 23


def test_fit_transform(sample_features_target):
    """Test scaler fit and transform."""
    X, _ = sample_features_target
    pp = FraudPreprocessor()
    X_scaled = pp.fit_transform(X)

    assert pp._fitted is True
    assert X_scaled.shape == X.shape
    # After StandardScaler, mean should be ~0 and std ~1
    assert abs(X_scaled["V1"].mean()) < 0.1
    assert abs(X_scaled["V1"].std() - 1.0) < 0.1


def test_transform_before_fit(sample_features_target):
    """Test that transform fails before fit."""
    X, _ = sample_features_target
    pp = FraudPreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        pp.transform(X)
