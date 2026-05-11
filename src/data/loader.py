"""Data loading utilities for fraud detection datasets."""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_creditcard(path: str = None) -> pd.DataFrame:
    """Load the Kaggle Credit Card Fraud Detection dataset.

    Args:
        path: Path to creditcard.csv. Uses default from config if None.

    Returns:
        DataFrame with 284,807 rows and 31 columns.
    """
    if path is None:
        from config import KAGGLE_CC_PATH
        path = KAGGLE_CC_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    df = pd.read_csv(path)

    expected_shape_cols = 31
    if df.shape[1] != expected_shape_cols:
        raise ValueError(
            f"Expected {expected_shape_cols} columns, got {df.shape[1]}. "
            "Check dataset integrity."
        )

    logger.info(
        f"Loaded Credit Card dataset: {df.shape[0]:,} rows, {df.shape[1]} columns. "
        f"Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)"
    )
    return df


def load_paysim(path: str = None) -> pd.DataFrame:
    """Load the PaySim synthetic financial dataset.

    Args:
        path: Path to paysim CSV. Uses default from config if None.

    Returns:
        DataFrame with ~6.3M rows.
    """
    if path is None:
        from config import PAYSIM_PATH
        path = PAYSIM_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download from https://www.kaggle.com/datasets/ealaxi/paysim1"
        )

    df = pd.read_csv(path)

    required_cols = {"type", "amount", "oldbalanceOrg", "newbalanceOrig",
                     "oldbalanceDest", "newbalanceDest", "isFraud"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in PaySim dataset: {missing}")

    logger.info(
        f"Loaded PaySim dataset: {df.shape[0]:,} rows, {df.shape[1]} columns. "
        f"Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)"
    )
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary statistics about a dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with shape, dtypes, null counts, and class distribution.
    """
    target_col = "Class" if "Class" in df.columns else "isFraud"

    info = {
        "shape": df.shape,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": df.dtypes.value_counts().to_dict(),
        "null_counts": df.isnull().sum().sum(),
        "null_per_column": df.isnull().sum().to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }

    if target_col in df.columns:
        class_counts = df[target_col].value_counts()
        info["class_distribution"] = class_counts.to_dict()
        info["fraud_rate"] = df[target_col].mean()
        info["imbalance_ratio"] = (
            class_counts.get(0, 0) / max(class_counts.get(1, 1), 1)
        )

    return info


def validate_dataset(df: pd.DataFrame, expected_cols: list) -> bool:
    """Verify that expected columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        expected_cols: List of column names to check.

    Returns:
        True if all columns present, raises ValueError otherwise.
    """
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return True
