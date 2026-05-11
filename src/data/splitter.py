"""Stratified data splitting for imbalanced classification."""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)


def stratified_split(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE,
                     random_state=RANDOM_STATE):
    """Split data into train/val/test sets preserving class ratios.

    First splits into (train+val) / test, then splits train+val into train / val.

    Args:
        X: Features (DataFrame or array).
        y: Target (Series or array).
        test_size: Proportion of data for test set.
        val_size: Proportion of remaining data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=random_state
    )

    stats = get_split_stats(y_train, y_val, y_test)
    logger.info(
        f"Split complete — Train: {len(y_train):,} "
        f"(fraud: {stats['train']['fraud_count']:,}, {stats['train']['fraud_rate']:.3f}%) | "
        f"Val: {len(y_val):,} "
        f"(fraud: {stats['val']['fraud_count']:,}, {stats['val']['fraud_rate']:.3f}%) | "
        f"Test: {len(y_test):,} "
        f"(fraud: {stats['test']['fraud_count']:,}, {stats['test']['fraud_rate']:.3f}%)"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_split_stats(y_train, y_val, y_test) -> dict:
    """Compute class distribution statistics for each split.

    Args:
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.

    Returns:
        Dictionary with fraud counts and rates per split.
    """
    def _stats(y):
        total = len(y)
        fraud = int(sum(y == 1)) if hasattr(y, '__iter__') else 0
        return {
            "total": total,
            "fraud_count": fraud,
            "legit_count": total - fraud,
            "fraud_rate": (fraud / total * 100) if total > 0 else 0,
        }

    return {
        "train": _stats(y_train),
        "val": _stats(y_val),
        "test": _stats(y_test),
    }
