"""Hyperparameter optimization using Optuna."""

import logging
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaHyperTuner:
    """Optuna-based hyperparameter optimization for fraud detection models."""

    def __init__(self, model_name: str, X_train, y_train, X_val=None, y_val=None,
                 scoring: str = "roc_auc", cv_folds: int = CV_FOLDS):
        """Initialize the tuner.

        Args:
            model_name: Name of model to tune.
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features (unused for CV-based tuning).
            y_val: Optional validation labels.
            scoring: Optimization metric.
            cv_folds: Number of cross-validation folds.
        """
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=RANDOM_STATE)
        self.study = None
        self.best_params = None

    def _get_search_space(self, trial):
        """Define hyperparameter search space per model.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of hyperparameters for this trial.
        """
        if self.model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            }

        elif self.model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "eval_metric": "auc",
                "use_label_encoder": False,
                "random_state": RANDOM_STATE,
            }

        elif self.model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "is_unbalance": True,
                "verbose": -1,
                "random_state": RANDOM_STATE,
            }

        else:
            raise ValueError(f"No search space defined for model: {self.model_name}")

    def _create_model(self, params):
        """Instantiate a model with given parameters."""
        if self.model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif self.model_name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        elif self.model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        else:
            raise ValueError(f"Cannot create model: {self.model_name}")

    def objective(self, trial):
        """Optuna objective function.

        Args:
            trial: Optuna trial.

        Returns:
            Mean cross-validation score.
        """
        params = self._get_search_space(trial)
        model = self._create_model(params)
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv, scoring=self.scoring, n_jobs=-1
        )
        return scores.mean()

    def tune(self, n_trials: int = 50) -> dict:
        """Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run.

        Returns:
            Best hyperparameters found.
        """
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)

        self.best_params = self.study.best_params
        logger.info(
            f"Optuna tuning for {self.model_name} complete. "
            f"Best {self.scoring}: {self.study.best_value:.4f}. "
            f"Best params: {self.best_params}"
        )
        return self.best_params

    def get_best_params(self) -> dict:
        """Return the optimal hyperparameters."""
        if self.best_params is None:
            raise RuntimeError("Call tune() first.")
        return self.best_params

    def get_study_results(self) -> list:
        """Return full optimization history as list of dicts."""
        if self.study is None:
            raise RuntimeError("Call tune() first.")
        return [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in self.study.trials
        ]
