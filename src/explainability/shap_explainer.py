"""SHAP-based explainability for fraud detection models."""

import os
import logging
import numpy as np
import shap
import matplotlib.pyplot as plt

from config import FIGURES_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP global and local explanations for fraud detection models."""

    def __init__(self, model, X_background, feature_names: list = None,
                 use_tree: bool = True):
        """Initialize SHAP explainer.

        Args:
            model: Trained model instance.
            X_background: Background dataset for SHAP (use shap.sample for speed).
            feature_names: List of feature names.
            use_tree: Use TreeExplainer (True) or KernelExplainer (False).
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None

        if use_tree:
            try:
                self.explainer = shap.TreeExplainer(model)
                logger.info("Initialized SHAP TreeExplainer")
            except Exception:
                logger.info("TreeExplainer failed, falling back to KernelExplainer")
                background = shap.sample(X_background, 100)
                predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                self.explainer = shap.KernelExplainer(predict_fn, background)
        else:
            background = shap.sample(X_background, 100)
            predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            self.explainer = shap.KernelExplainer(predict_fn, background)
            logger.info("Initialized SHAP KernelExplainer")

    def compute_shap_values(self, X):
        """Calculate SHAP values for a dataset.

        Args:
            X: Feature matrix.

        Returns:
            SHAP values array.
        """
        self.shap_values = self.explainer.shap_values(X)
        # For binary classification, take the fraud class values
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]
        logger.info(f"Computed SHAP values for {X.shape[0]} samples")
        return self.shap_values

    def plot_global_importance(self, X, top_n: int = 10, save_path: str = None):
        """Bar chart of mean absolute SHAP values (top features).

        Args:
            X: Feature matrix used for SHAP computation.
            top_n: Number of top features to show.
            save_path: Save path.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=top_n,
            show=False
        )
        plt.title("Importance Globale des Features (SHAP)")
        plt.tight_layout()

        path = save_path or os.path.join(FIGURES_DIR, "explainability", "shap_global.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"SHAP global importance plot saved to {path}")

    def plot_summary(self, X, save_path: str = None):
        """Beeswarm summary plot (feature × SHAP × value).

        Args:
            X: Feature matrix.
            save_path: Save path.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Summary Plot (Beeswarm)")
        plt.tight_layout()

        path = save_path or os.path.join(FIGURES_DIR, "explainability", "shap_summary.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"SHAP summary plot saved to {path}")

    def plot_force_single(self, X, idx: int, save_path: str = None):
        """Force plot for a single prediction.

        Args:
            X: Feature matrix.
            idx: Index of the sample to explain.
            save_path: Save path.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

        fig = shap.force_plot(
            expected_value,
            self.shap_values[idx],
            X.iloc[idx] if hasattr(X, 'iloc') else X[idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        path = save_path or os.path.join(FIGURES_DIR, "explainability", f"shap_force_{idx}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"SHAP force plot for sample {idx} saved to {path}")

    def plot_dependence(self, X, feature: str, save_path: str = None):
        """Dependence plot for a single feature.

        Args:
            X: Feature matrix.
            feature: Feature name.
            save_path: Save path.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature, self.shap_values, X,
            feature_names=self.feature_names,
            show=False, ax=ax
        )
        plt.tight_layout()

        safe_name = feature.replace(" ", "_").lower()
        path = save_path or os.path.join(FIGURES_DIR, "explainability", f"shap_dep_{safe_name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()

    def get_top_features(self, n: int = 10) -> list:
        """Return names of the most important features.

        Args:
            n: Number of top features.

        Returns:
            List of feature names sorted by importance.
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[::-1][:n]

        if self.feature_names:
            return [self.feature_names[i] for i in indices]
        return indices.tolist()

    def save_plots(self, X, output_dir: str = None):
        """Generate and save all SHAP plots.

        Args:
            X: Feature matrix.
            output_dir: Output directory.
        """
        if output_dir is None:
            output_dir = os.path.join(FIGURES_DIR, "explainability")
        os.makedirs(output_dir, exist_ok=True)

        self.plot_global_importance(X, save_path=os.path.join(output_dir, "shap_global.png"))
        self.plot_summary(X, save_path=os.path.join(output_dir, "shap_summary.png"))

        # Force plots for a few samples
        for i in range(min(3, len(X))):
            self.plot_force_single(X, i, save_path=os.path.join(output_dir, f"shap_force_{i}.png"))

        logger.info(f"All SHAP plots saved to {output_dir}")
