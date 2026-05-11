"""Deep learning models for fraud detection (MLP and Autoencoder)."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MLPFraudClassifier:
    """Multi-Layer Perceptron for binary fraud classification.

    Architecture: Input → Dense(128) → Dropout → Dense(64) → Dropout
                  → Dense(32) → Dropout → Dense(1, sigmoid)
    """

    def __init__(self, input_dim: int, hidden_layers: list = None,
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """Initialize MLP architecture.

        Args:
            input_dim: Number of input features.
            hidden_layers: List of hidden layer sizes. Default [128, 64, 32].
            dropout_rate: Dropout rate between layers.
            learning_rate: Adam optimizer learning rate.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self._build_model()

    def _build_model(self):
        """Build the Keras Sequential model."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))

        for i, units in enumerate(self.hidden_layers):
            model.add(Dense(units, activation="relu"))
            # Reduce dropout for the last hidden layer
            dr = self.dropout_rate if i < len(self.hidden_layers) - 1 else self.dropout_rate * 0.67
            model.add(Dropout(dr))

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )

        self.model = model
        logger.info(f"MLP built: {self.hidden_layers}, dropout={self.dropout_rate}")

    def fit(self, X, y, epochs: int = 50, batch_size: int = 256,
            validation_data=None, class_weight: dict = None):
        """Train the MLP.

        Args:
            X: Training features.
            y: Training labels.
            epochs: Max training epochs.
            batch_size: Batch size.
            validation_data: Tuple of (X_val, y_val).
            class_weight: Class weight dictionary for imbalance handling.

        Returns:
            Training history.
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor="val_auc" if validation_data else "auc",
                patience=5,
                restore_best_weights=True,
                mode="max"
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        return self.history

    def predict(self, X) -> np.ndarray:
        """Return fraud probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of fraud probabilities.
        """
        return self.model.predict(X, verbose=0).flatten()

    def predict_classes(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at given threshold.

        Args:
            X: Feature matrix.
            threshold: Classification threshold.

        Returns:
            Binary array (0 or 1).
        """
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def save(self, path: str):
        """Save model to disk."""
        self.model.save(path)
        logger.info(f"MLP model saved to {path}")

    @classmethod
    def load(cls, path: str, input_dim: int = 30):
        """Load model from disk."""
        import tensorflow as tf
        instance = cls.__new__(cls)
        instance.model = tf.keras.models.load_model(path)
        instance.input_dim = input_dim
        return instance


class AutoencoderDetector:
    """Autoencoder-based anomaly detector for fraud detection.

    Trained on legitimate transactions only. High reconstruction error
    indicates anomalous (potentially fraudulent) transactions.

    Architecture: Encoder: Input(n) → Dense(16) → Dense(8)
                  Decoder: Dense(16) → Dense(n)
    """

    def __init__(self, input_dim: int, encoding_dim: int = 8,
                 hidden_dims: list = None, learning_rate: float = 0.001):
        """Initialize autoencoder.

        Args:
            input_dim: Number of input features.
            encoding_dim: Size of the bottleneck layer.
            hidden_dims: Encoder hidden layer sizes. Default [16].
            learning_rate: Adam optimizer learning rate.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [16]
        self.learning_rate = learning_rate
        self.model = None
        self.threshold = None
        self._build_model()

    def _build_model(self):
        """Build the autoencoder architecture."""
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.optimizers import Adam

        input_layer = Input(shape=(self.input_dim,))

        # Encoder
        x = input_layer
        for dim in self.hidden_dims:
            x = Dense(dim, activation="relu")(x)
        encoded = Dense(self.encoding_dim, activation="relu")(x)

        # Decoder (mirror of encoder)
        x = encoded
        for dim in reversed(self.hidden_dims):
            x = Dense(dim, activation="relu")(x)
        decoded = Dense(self.input_dim, activation="sigmoid")(x)

        self.model = Model(input_layer, decoded)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        logger.info(
            f"Autoencoder built: {self.input_dim} → "
            f"{self.hidden_dims} → {self.encoding_dim} → "
            f"{list(reversed(self.hidden_dims))} → {self.input_dim}"
        )

    def fit(self, X_normal, epochs: int = 50, batch_size: int = 256,
            validation_split: float = 0.1):
        """Train on legitimate transactions only.

        Args:
            X_normal: Feature matrix of LEGITIMATE transactions only.
            epochs: Max training epochs.
            batch_size: Batch size.
            validation_split: Fraction of data for validation.

        Returns:
            Training history.
        """
        from tensorflow.keras.callbacks import EarlyStopping

        history = self.model.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
        return history

    def compute_reconstruction_error(self, X) -> np.ndarray:
        """Compute MSE reconstruction error for each sample.

        Args:
            X: Feature matrix.

        Returns:
            Array of reconstruction errors (MSE per sample).
        """
        X_pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
        return mse

    def find_optimal_threshold(self, X_val, y_val) -> float:
        """Find the threshold maximizing F1-score on validation data.

        Args:
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Optimal threshold value.
        """
        from sklearn.metrics import f1_score

        errors = self.compute_reconstruction_error(X_val)
        thresholds = np.percentile(errors, np.arange(80, 100, 0.5))

        best_f1 = 0
        best_threshold = thresholds[0]

        for t in thresholds:
            preds = (errors > t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        self.threshold = best_threshold
        logger.info(f"Optimal AE threshold: {best_threshold:.6f} (F1={best_f1:.4f})")
        return best_threshold

    def predict(self, X, threshold: float = None) -> np.ndarray:
        """Predict fraud based on reconstruction error threshold.

        Args:
            X: Feature matrix.
            threshold: Error threshold. Uses fitted optimal if None.

        Returns:
            Binary array (1 = fraud / anomaly).
        """
        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            raise ValueError("No threshold set. Call find_optimal_threshold() first.")

        errors = self.compute_reconstruction_error(X)
        return (errors > threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Return anomaly scores (normalized reconstruction errors).

        Args:
            X: Feature matrix.

        Returns:
            Array of anomaly scores (higher = more anomalous).
        """
        errors = self.compute_reconstruction_error(X)
        # Normalize to [0, 1] range using min-max
        if errors.max() > errors.min():
            normalized = (errors - errors.min()) / (errors.max() - errors.min())
        else:
            normalized = np.zeros_like(errors)
        return normalized

    def save(self, path: str):
        """Save model to disk."""
        self.model.save(path)
        logger.info(f"Autoencoder saved to {path}")

    @classmethod
    def load(cls, path: str, input_dim: int = 30):
        """Load model from disk."""
        import tensorflow as tf
        instance = cls.__new__(cls)
        instance.model = tf.keras.models.load_model(path)
        instance.input_dim = input_dim
        instance.threshold = None
        return instance
