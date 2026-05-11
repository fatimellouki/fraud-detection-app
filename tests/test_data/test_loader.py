"""Tests for data loading module."""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import get_dataset_info, validate_dataset


def test_get_dataset_info(sample_kaggle_data):
    """Test dataset info extraction."""
    info = get_dataset_info(sample_kaggle_data)

    assert info["n_rows"] == 1017
    assert info["n_cols"] == 31
    assert info["null_counts"] == 0
    assert info["fraud_rate"] > 0
    assert "class_distribution" in info
    assert info["class_distribution"][0] == 1000
    assert info["class_distribution"][1] == 17


def test_validate_dataset_success(sample_kaggle_data):
    """Test validation with correct columns."""
    expected_cols = ["V1", "V2", "Amount", "Time", "Class"]
    assert validate_dataset(sample_kaggle_data, expected_cols) is True


def test_validate_dataset_missing_columns(sample_kaggle_data):
    """Test validation with missing columns."""
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_dataset(sample_kaggle_data, ["V1", "NonExistentColumn"])
