"""Stratified K-Fold cross-validation for fraud detection."""

import time
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from src.evaluation.metrics import compute_all_metrics
from config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)


class StratifiedCrossValidator:
    """Run stratified K-fold cross-validation and aggregate metrics."""

    def __init__(self, n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        """Initialize cross-validator.

        Args:
            n_splits: Number of folds.
            random_state: Random seed.
        """
        self.n_splits = n_splits
        self.kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        self.fold_results = []

    def validate(self, model, X, y) -> list:
        """Run k-fold cross-validation.

        Args:
            model: Sklearn-compatible model instance.
            X: Feature matrix.
            y: Target vector.

        Returns:
            List of per-fold metric dictionaries.
        """
        self.fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X, y), 1):
            if isinstance(X, np.ndarray):
                X_train, X_val = X[train_idx], X[val_idx]
            else:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

            if isinstance(y, np.ndarray):
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone model to avoid data leakage between folds
            fold_model = clone(model)

            start = time.time()
            fold_model.fit(X_train, y_train)
            train_time = time.time() - start

            # Predictions
            if hasattr(fold_model, "predict_proba"):
                y_proba = fold_model.predict_proba(X_val)[:, 1]
            else:
                y_proba = fold_model.predict(X_val).flatten()

            y_pred = (y_proba >= 0.5).astype(int)

            # Metrics
            metrics = compute_all_metrics(y_val, y_pred, y_proba)
            metrics["fold"] = fold_idx
            metrics["train_time_s"] = round(train_time, 2)

            # Inference time
            start = time.time()
            if hasattr(fold_model, "predict_proba"):
                fold_model.predict_proba(X_val[:100])
            else:
                fold_model.predict(X_val[:100])
            inf_time = (time.time() - start) / min(100, len(X_val)) * 1000
            metrics["inference_time_ms"] = round(inf_time, 4)

            self.fold_results.append(metrics)
            logger.info(
                f"  Fold {fold_idx}/{self.n_splits}: "
                f"AUC={metrics['auc_roc']:.4f}, F1={metrics['f1_score']:.4f}"
            )

        return self.fold_results

    def get_mean_scores(self) -> dict:
        """Return mean and std for all metrics across folds.

        Returns:
            Dictionary with 'metric_mean' and 'metric_std' keys.
        """
        if not self.fold_results:
            raise RuntimeError("Call validate() first.")

        metric_keys = [k for k in self.fold_results[0]
                       if k not in ("fold", "confusion_matrix", "model_name")]

        summary = {}
        for key in metric_keys:
            values = [f[key] for f in self.fold_results if isinstance(f.get(key), (int, float))]
            if values:
                summary[f"{key}_mean"] = round(np.mean(values), 4)
                summary[f"{key}_std"] = round(np.std(values), 4)

        return summary

    def get_fold_details(self) -> list:
        """Return detailed per-fold results."""
        return self.fold_results
