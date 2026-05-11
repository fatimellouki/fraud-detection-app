"""Class imbalance handling techniques for fraud detection."""

import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from config import RANDOM_STATE

logger = logging.getLogger(__name__)


class ImbalanceHandler:
    """Applies various class imbalance techniques and compares results."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def apply_smote(self, X, y, sampling_strategy=0.5, k_neighbors=5):
        """Apply SMOTE oversampling.

        Args:
            X: Feature matrix.
            y: Target vector.
            sampling_strategy: Ratio of minority to majority after resampling.
            k_neighbors: Number of nearest neighbors for interpolation.

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(
            f"SMOTE: {len(y)} → {len(y_res)} samples "
            f"(fraud: {sum(y==1)} → {sum(y_res==1)})"
        )
        return X_res, y_res

    def apply_adasyn(self, X, y, sampling_strategy=0.5):
        """Apply ADASYN adaptive oversampling.

        Args:
            X: Feature matrix.
            y: Target vector.
            sampling_strategy: Desired ratio after resampling.

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_res, y_res = adasyn.fit_resample(X, y)
        logger.info(
            f"ADASYN: {len(y)} → {len(y_res)} samples "
            f"(fraud: {sum(y==1)} → {sum(y_res==1)})"
        )
        return X_res, y_res

    def apply_random_undersampling(self, X, y, sampling_strategy=0.5):
        """Apply random undersampling of majority class.

        Args:
            X: Feature matrix.
            y: Target vector.
            sampling_strategy: Desired ratio after undersampling.

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_res, y_res = rus.fit_resample(X, y)
        logger.info(
            f"Random Undersampling: {len(y)} → {len(y_res)} samples "
            f"(legit: {sum(y==0)} → {sum(y_res==0)})"
        )
        return X_res, y_res

    def apply_smote_tomek(self, X, y, sampling_strategy=0.5):
        """Apply SMOTE combined with Tomek links cleaning.

        Args:
            X: Feature matrix.
            y: Target vector.
            sampling_strategy: Desired ratio for SMOTE step.

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        smt = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            ),
            random_state=self.random_state
        )
        X_res, y_res = smt.fit_resample(X, y)
        logger.info(
            f"SMOTE+Tomek: {len(y)} → {len(y_res)} samples "
            f"(fraud: {sum(y==1)} → {sum(y_res==1)})"
        )
        return X_res, y_res

    @staticmethod
    def get_class_weights(y) -> dict:
        """Compute balanced class weights for cost-sensitive learning.

        Args:
            y: Target vector.

        Returns:
            Dictionary mapping class labels to weights.
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
        logger.info(f"Class weights: {weight_dict}")
        return weight_dict

    def compare_techniques(self, X, y) -> dict:
        """Run all imbalance techniques and return statistics.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Dictionary mapping technique name to (n_samples, n_fraud, n_legit).
        """
        results = {}

        # No resampling (baseline)
        results["none"] = {
            "n_samples": len(y),
            "n_fraud": int(sum(y == 1)),
            "n_legit": int(sum(y == 0)),
        }

        techniques = [
            ("smote", self.apply_smote),
            ("adasyn", self.apply_adasyn),
            ("undersampling", self.apply_random_undersampling),
            ("smote_tomek", self.apply_smote_tomek),
        ]

        for name, func in techniques:
            try:
                X_res, y_res = func(X, y)
                results[name] = {
                    "n_samples": len(y_res),
                    "n_fraud": int(sum(y_res == 1)),
                    "n_legit": int(sum(y_res == 0)),
                }
            except Exception as e:
                logger.warning(f"Technique {name} failed: {e}")
                results[name] = {"error": str(e)}

        # Cost-sensitive (just report weights, no resampling)
        weights = self.get_class_weights(y)
        results["cost_sensitive"] = {
            "n_samples": len(y),
            "n_fraud": int(sum(y == 1)),
            "n_legit": int(sum(y == 0)),
            "class_weights": weights,
        }

        return results
