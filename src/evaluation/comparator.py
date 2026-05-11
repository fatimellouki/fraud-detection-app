"""Model comparison utilities for side-by-side evaluation."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class ModelComparator:
    """Collect and compare results across multiple models."""

    def __init__(self):
        self.results = {}

    def add_result(self, model_name: str, metrics_dict: dict):
        """Store metrics for a model.

        Args:
            model_name: Identifier for the model.
            metrics_dict: Dictionary of metric values.
        """
        self.results[model_name] = metrics_dict
        logger.info(f"Added results for {model_name}")

    def get_comparison_table(self) -> pd.DataFrame:
        """Return DataFrame with all models and metrics.

        Returns:
            DataFrame with model names as index.
        """
        df = pd.DataFrame(self.results).T
        # Order columns for readability
        priority_cols = [
            "auc_roc", "auprc", "f1_score", "precision", "recall",
            "accuracy", "specificity", "train_time_s", "inference_time_ms"
        ]
        ordered = [c for c in priority_cols if c in df.columns]
        remaining = [c for c in df.columns if c not in ordered]
        return df[ordered + remaining]

    def get_best_model(self, metric: str = "auc_roc") -> str:
        """Return the name of the best model for a given metric.

        Args:
            metric: Metric to rank by.

        Returns:
            Model name with highest score.
        """
        df = self.get_comparison_table()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        return df[metric].astype(float).idxmax()

    def get_ranking(self, metric: str = "auc_roc") -> pd.Series:
        """Return models ranked by a metric (descending).

        Args:
            metric: Metric to rank by.

        Returns:
            Sorted Series.
        """
        df = self.get_comparison_table()
        return df[metric].astype(float).sort_values(ascending=False)

    def export_latex_table(self, metrics: list = None) -> str:
        """Export comparison table as LaTeX.

        Args:
            metrics: Columns to include. Defaults to main metrics.

        Returns:
            LaTeX table string.
        """
        if metrics is None:
            metrics = ["auc_roc", "auprc", "f1_score", "precision",
                       "recall", "inference_time_ms"]

        df = self.get_comparison_table()
        available = [m for m in metrics if m in df.columns]
        return df[available].to_latex(float_format="%.4f")

    def export_csv(self, path: str):
        """Save comparison results to CSV.

        Args:
            path: Output file path.
        """
        df = self.get_comparison_table()
        df.to_csv(path)
        logger.info(f"Results exported to {path}")
