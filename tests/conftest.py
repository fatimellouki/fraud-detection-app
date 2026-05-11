"""Pytest fixtures for fraud detection tests."""

import numpy as np
import pandas as pd
import pytest

from config import RANDOM_STATE


@pytest.fixture
def sample_kaggle_data():
    """Generate synthetic data resembling the Kaggle CC dataset."""
    np.random.seed(RANDOM_STATE)
    n_legit = 1000
    n_fraud = 17  # ~1.7% to keep it manageable for tests

    # PCA features V1-V28
    X_legit = np.random.randn(n_legit, 28)
    X_fraud = np.random.randn(n_fraud, 28) * 1.5 + 0.5  # Slightly different distribution

    # Amount and Time
    amount_legit = np.random.exponential(80, n_legit)
    amount_fraud = np.random.exponential(120, n_fraud)
    time_legit = np.random.uniform(0, 172800, n_legit)
    time_fraud = np.random.uniform(0, 172800, n_fraud)

    X = np.vstack([
        np.column_stack([X_legit, amount_legit, time_legit]),
        np.column_stack([X_fraud, amount_fraud, time_fraud]),
    ])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    df = pd.DataFrame(X, columns=columns)
    df["Class"] = y.astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


@pytest.fixture
def sample_features_target(sample_kaggle_data):
    """Return X, y split from sample data."""
    df = sample_kaggle_data
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


@pytest.fixture
def small_train_test(sample_features_target):
    """Return a small train/test split."""
    from sklearn.model_selection import train_test_split
    X, y = sample_features_target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
