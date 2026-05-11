"""Baseline ML models for fraud detection."""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import LOGISTIC_REGRESSION_PARAMS, DECISION_TREE_PARAMS


def create_logistic_regression(**kwargs) -> LogisticRegression:
    """Create a Logistic Regression classifier with default or custom params.

    Args:
        **kwargs: Override default hyperparameters.

    Returns:
        Configured LogisticRegression instance.
    """
    params = {**LOGISTIC_REGRESSION_PARAMS, **kwargs}
    return LogisticRegression(**params)


def create_decision_tree(**kwargs) -> DecisionTreeClassifier:
    """Create a Decision Tree classifier with default or custom params.

    Args:
        **kwargs: Override default hyperparameters.

    Returns:
        Configured DecisionTreeClassifier instance.
    """
    params = {**DECISION_TREE_PARAMS, **kwargs}
    return DecisionTreeClassifier(**params)
