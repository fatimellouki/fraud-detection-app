"""Stacking Ensemble model — main contribution of the project."""

import logging
import joblib
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

from src.models.ensemble_models import (
    create_random_forest, create_xgboost, create_lightgbm
)
from config import STACKING_META_PARAMS, CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)


class FraudStackingEnsemble:
    """Two-level stacking ensemble for fraud detection.

    Level 0 (Base Learners):
        - RandomForest: captures interactions via bagging
        - XGBoost: gradient boosting with regularization
        - LightGBM: fast leaf-wise gradient boosting

    Level 1 (Meta-Learner):
        - XGBoost (shallow): combines base predictions

    Cross-validation: StratifiedKFold to generate level-0 predictions
    while preserving class ratios.
    """

    def __init__(self, rf_params: dict = None, xgb_params: dict = None,
                 lgbm_params: dict = None, meta_params: dict = None,
                 cv_folds: int = CV_FOLDS):
        """Configure the stacking architecture.

        Args:
            rf_params: Override Random Forest hyperparameters.
            xgb_params: Override XGBoost hyperparameters.
            lgbm_params: Override LightGBM hyperparameters.
            meta_params: Override meta-learner hyperparameters.
            cv_folds: Number of folds for generating level-0 predictions.
        """
        self.rf_params = rf_params or {}
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.meta_params = meta_params or STACKING_META_PARAMS
        self.cv_folds = cv_folds
        self.model = None
        self._build()

    def _build(self):
        """Create the StackingClassifier from scikit-learn."""
        from xgboost import XGBClassifier

        base_estimators = [
            ("rf", create_random_forest(**self.rf_params)),
            ("xgb", create_xgboost(**self.xgb_params)),
            ("lgbm", create_lightgbm(**self.lgbm_params)),
        ]

        meta_learner = XGBClassifier(**self.meta_params)

        self.model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=RANDOM_STATE),
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )

        logger.info(
            f"Stacking ensemble built: "
            f"Base=[RF, XGBoost, LightGBM], "
            f"Meta=XGBoost(depth={self.meta_params.get('max_depth', 3)}), "
            f"CV={self.cv_folds}-fold"
        )

    def fit(self, X_train, y_train):
        """Train the stacking ensemble.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            self
        """
        logger.info("Training stacking ensemble (this may take a while)...")
        self.model.fit(X_train, y_train)
        logger.info("Stacking ensemble training complete.")
        return self

    def predict(self, X) -> np.ndarray:
        """Get binary predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of binary predictions.
        """
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Get fraud probability scores.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [P(legit), P(fraud)].
        """
        return self.model.predict_proba(X)

    def get_base_predictions(self, X) -> dict:
        """Get level-0 predictions from each base model.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping model name to probability predictions.
        """
        predictions = {}
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, "predict_proba"):
                predictions[name] = estimator.predict_proba(X)[:, 1]
            else:
                predictions[name] = estimator.predict(X)
        return predictions

    def save(self, path: str):
        """Serialize the entire ensemble to disk.

        Args:
            path: Output file path (.pkl).
        """
        joblib.dump(self.model, path)
        logger.info(f"Stacking ensemble saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load a saved stacking ensemble.

        Args:
            path: Path to saved model file.

        Returns:
            FraudStackingEnsemble instance with loaded model.
        """
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        logger.info(f"Stacking ensemble loaded from {path}")
        return instance
