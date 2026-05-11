"""Unified training loop for all model types."""

import time
import logging
import numpy as np
import joblib

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Unified trainer: fit, evaluate, time, and persist any model."""

    def __init__(self, model, model_name: str):
        """Initialize trainer.

        Args:
            model: Model instance (sklearn-compatible or deep learning wrapper).
            model_name: Human-readable name for logging.
        """
        self.model = model
        self.model_name = model_name
        self._train_time = None
        self._metrics = None

    def train(self, X_train, y_train, X_val=None, y_val=None, **fit_kwargs):
        """Fit the model and track training time.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            **fit_kwargs: Extra arguments passed to model.fit().

        Returns:
            self
        """
        logger.info(f"Training {self.model_name}...")
        start = time.time()

        # Deep learning models handle validation differently
        is_deep = hasattr(self.model, "history") or self.model_name in ("mlp", "autoencoder")

        if is_deep and X_val is not None and y_val is not None:
            if self.model_name == "autoencoder":
                # Autoencoder trains on legitimate transactions only
                X_normal = X_train[y_train == 0]
                self.model.fit(X_normal, **fit_kwargs)
                if X_val is not None and y_val is not None:
                    self.model.find_optimal_threshold(X_val, y_val)
            else:
                self.model.fit(X_train, y_train,
                               validation_data=(X_val, y_val), **fit_kwargs)
        else:
            self.model.fit(X_train, y_train)

        self._train_time = time.time() - start
        logger.info(
            f"{self.model_name} trained in {self._train_time:.2f}s"
        )
        return self

    def evaluate(self, X_test, y_test) -> dict:
        """Compute all metrics on the test set.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary of metrics.
        """
        # Get predictions
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
        elif hasattr(self.model, "predict"):
            y_proba = self.model.predict(X_test)
            if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
        else:
            raise AttributeError(f"{self.model_name} has no predict method")

        # Binary predictions
        if hasattr(self.model, "predict_classes"):
            y_pred = self.model.predict_classes(X_test)
        else:
            y_pred = (y_proba >= 0.5).astype(int)

        self._metrics = compute_all_metrics(y_test, y_pred, y_proba)
        self._metrics["model_name"] = self.model_name
        self._metrics["train_time_s"] = self._train_time

        # Inference time
        self._metrics["inference_time_ms"] = self.get_inference_time(X_test[:100])

        logger.info(
            f"{self.model_name} — AUC-ROC: {self._metrics['auc_roc']:.4f}, "
            f"F1: {self._metrics['f1_score']:.4f}, "
            f"Precision: {self._metrics['precision']:.4f}, "
            f"Recall: {self._metrics['recall']:.4f}"
        )
        return self._metrics

    def get_training_time(self) -> float:
        """Return training duration in seconds."""
        return self._train_time

    def get_inference_time(self, X_sample, n_runs: int = 10) -> float:
        """Return average prediction time per sample in milliseconds.

        Args:
            X_sample: Sample feature matrix.
            n_runs: Number of runs to average over.

        Returns:
            Average ms per prediction.
        """
        times = []
        for _ in range(n_runs):
            start = time.time()
            if hasattr(self.model, "predict_proba"):
                self.model.predict_proba(X_sample)
            else:
                self.model.predict(X_sample)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_total = np.mean(times)
        n_samples = len(X_sample) if hasattr(X_sample, "__len__") else 1
        ms_per_sample = (avg_total / n_samples) * 1000
        return round(ms_per_sample, 4)

    def save_model(self, path: str):
        """Serialize the trained model.

        Args:
            path: Output file path.
        """
        if hasattr(self.model, "save"):
            self.model.save(path)
        else:
            joblib.dump(self.model, path)
        logger.info(f"{self.model_name} saved to {path}")
