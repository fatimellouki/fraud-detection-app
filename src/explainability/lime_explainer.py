"""LIME-based explainability for fraud detection models."""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

from config import FIGURES_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """LIME instance-level explanations for fraud detection."""

    def __init__(self, X_train, feature_names: list = None,
                 class_names: list = None):
        """Initialize LIME explainer.

        Args:
            X_train: Training data for computing feature statistics.
            feature_names: List of feature names.
            class_names: List of class names. Default: ["Légitime", "Fraude"].
        """
        self.feature_names = feature_names
        self.class_names = class_names or ["Légitime", "Fraude"]

        train_data = X_train.values if hasattr(X_train, "values") else np.array(X_train)

        self.explainer = LimeTabularExplainer(
            training_data=train_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        logger.info("Initialized LIME TabularExplainer")

    def explain_instance(self, X_single, model, num_features: int = 10):
        """Generate explanation for one transaction.

        Args:
            X_single: Single sample (1D array).
            model: Trained model with predict_proba method.
            num_features: Number of features to show.

        Returns:
            LIME Explanation object.
        """
        sample = X_single.values if hasattr(X_single, "values") else np.array(X_single)
        if sample.ndim > 1:
            sample = sample.flatten()

        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict

        explanation = self.explainer.explain_instance(
            sample,
            predict_fn,
            num_features=num_features,
            num_samples=5000
        )
        return explanation

    def plot_explanation(self, explanation, save_path: str = None):
        """Visualize feature contributions from LIME explanation.

        Args:
            explanation: LIME Explanation object.
            save_path: Save path for the figure.
        """
        fig = explanation.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)
        plt.title("Explication LIME — Contributions des Features")
        plt.tight_layout()

        path = save_path or os.path.join(FIGURES_DIR, "explainability", "lime_explanation.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"LIME explanation plot saved to {path}")

    def explain_fraud_example(self, X_test, y_test, model, save_path: str = None):
        """Find and explain a true positive (correctly detected fraud).

        Args:
            X_test: Test features.
            y_test: Test labels.
            model: Trained model.
            save_path: Save path.

        Returns:
            LIME Explanation object for a fraud example.
        """
        y_true = np.array(y_test)
        fraud_indices = np.where(y_true == 1)[0]

        if len(fraud_indices) == 0:
            logger.warning("No fraud samples found in test data.")
            return None

        # Find a correctly predicted fraud
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2:
                proba = proba[:, 1]
        else:
            proba = model.predict(X_test)

        for idx in fraud_indices:
            if proba[idx] >= 0.5:
                sample = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
                explanation = self.explain_instance(sample, model)

                path = save_path or os.path.join(FIGURES_DIR, "explainability", "lime_fraud.png")
                self.plot_explanation(explanation, save_path=path)
                logger.info(f"Explained fraud example at index {idx} (proba={proba[idx]:.4f})")
                return explanation

        # Fallback: use the highest-probability fraud sample
        best_idx = fraud_indices[np.argmax(proba[fraud_indices])]
        sample = X_test.iloc[best_idx] if hasattr(X_test, 'iloc') else X_test[best_idx]
        explanation = self.explain_instance(sample, model)
        path = save_path or os.path.join(FIGURES_DIR, "explainability", "lime_fraud.png")
        self.plot_explanation(explanation, save_path=path)
        return explanation

    def explain_legitimate_example(self, X_test, y_test, model, save_path: str = None):
        """Find and explain a true negative (correctly classified legitimate).

        Args:
            X_test: Test features.
            y_test: Test labels.
            model: Trained model.
            save_path: Save path.

        Returns:
            LIME Explanation object for a legitimate example.
        """
        y_true = np.array(y_test)
        legit_indices = np.where(y_true == 0)[0]

        if len(legit_indices) == 0:
            return None

        # Pick a confidently correct legitimate transaction
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2:
                proba = proba[:, 1]
        else:
            proba = model.predict(X_test)

        # Find one with very low fraud probability
        for idx in legit_indices[:100]:
            if proba[idx] < 0.1:
                sample = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
                explanation = self.explain_instance(sample, model)

                path = save_path or os.path.join(FIGURES_DIR, "explainability", "lime_legit.png")
                self.plot_explanation(explanation, save_path=path)
                logger.info(f"Explained legitimate example at index {idx} (proba={proba[idx]:.4f})")
                return explanation

        return None

    def compare_explanations(self, X_fraud, X_legit, model, save_path: str = None):
        """Side-by-side fraud vs legitimate explanations.

        Args:
            X_fraud: Single fraud sample.
            X_legit: Single legitimate sample.
            model: Trained model.
            save_path: Save path.
        """
        exp_fraud = self.explain_instance(X_fraud, model)
        exp_legit = self.explain_instance(X_legit, model)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Fraud explanation
        fraud_weights = exp_fraud.as_list(label=1)
        features_f = [w[0] for w in fraud_weights[:8]]
        values_f = [w[1] for w in fraud_weights[:8]]
        colors_f = ["#e74c3c" if v > 0 else "#2ecc71" for v in values_f]
        axes[0].barh(features_f, values_f, color=colors_f)
        axes[0].set_title("Transaction Frauduleuse")
        axes[0].set_xlabel("Contribution")

        # Legitimate explanation
        legit_weights = exp_legit.as_list(label=1)
        features_l = [w[0] for w in legit_weights[:8]]
        values_l = [w[1] for w in legit_weights[:8]]
        colors_l = ["#e74c3c" if v > 0 else "#2ecc71" for v in values_l]
        axes[1].barh(features_l, values_l, color=colors_l)
        axes[1].set_title("Transaction Légitime")
        axes[1].set_xlabel("Contribution")

        fig.suptitle("Comparaison LIME: Fraude vs Légitime", fontsize=14)
        plt.tight_layout()

        path = save_path or os.path.join(FIGURES_DIR, "explainability", "lime_comparison.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"LIME comparison plot saved to {path}")

    def save_explanation_html(self, explanation, path: str):
        """Save interactive HTML explanation.

        Args:
            explanation: LIME Explanation object.
            path: Output HTML file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        explanation.save_to_file(path)
        logger.info(f"LIME HTML explanation saved to {path}")
