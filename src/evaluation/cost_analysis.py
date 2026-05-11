"""Cost-benefit analysis for fraud detection models."""

import numpy as np
import matplotlib.pyplot as plt
import logging

from config import AVG_FRAUD_AMOUNT, FP_COST, INVESTIGATION_COST, FIGURES_DIR
from src.evaluation.visualizer import save_figure

logger = logging.getLogger(__name__)


class CostBenefitAnalyzer:
    """Estimate financial impact of model predictions."""

    def __init__(self, avg_fraud_amount: float = AVG_FRAUD_AMOUNT,
                 fp_cost: float = FP_COST,
                 investigation_cost: float = INVESTIGATION_COST):
        """Set cost parameters.

        Args:
            avg_fraud_amount: Average monetary value of a fraudulent transaction.
            fp_cost: Cost per false positive (customer service).
            investigation_cost: Cost to review each flagged transaction.
        """
        self.avg_fraud_amount = avg_fraud_amount
        self.fp_cost = fp_cost
        self.investigation_cost = investigation_cost

    def analyze(self, y_true, y_pred, y_proba=None) -> dict:
        """Compute full cost-benefit analysis.

        Args:
            y_true: Ground truth labels.
            y_pred: Binary predictions.
            y_proba: Optional probability scores (unused in basic analysis).

        Returns:
            Dictionary with cost/benefit breakdown.
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        savings = self.compute_savings(tp, fn)
        fp_costs = self.compute_fp_cost(fp)
        investigation = (tp + fp) * self.investigation_cost

        net_benefit = savings["fraud_caught_value"] - savings["fraud_missed_value"] \
                      - fp_costs["total_fp_cost"] - investigation

        result = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fraud_caught_value": savings["fraud_caught_value"],
            "fraud_missed_value": savings["fraud_missed_value"],
            "fp_cost_total": fp_costs["total_fp_cost"],
            "investigation_cost_total": investigation,
            "net_benefit": round(net_benefit, 2),
            "detection_rate": round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0,
        }

        logger.info(
            f"Cost analysis: caught €{savings['fraud_caught_value']:.0f}, "
            f"missed €{savings['fraud_missed_value']:.0f}, "
            f"FP cost €{fp_costs['total_fp_cost']:.0f}, "
            f"net €{net_benefit:.0f}"
        )
        return result

    def compute_savings(self, tp: int, fn: int) -> dict:
        """Calculate money saved and lost.

        Args:
            tp: True positives (fraud correctly caught).
            fn: False negatives (fraud missed).

        Returns:
            Dictionary with financial values.
        """
        return {
            "fraud_caught_value": round(tp * self.avg_fraud_amount, 2),
            "fraud_missed_value": round(fn * self.avg_fraud_amount, 2),
        }

    def compute_fp_cost(self, fp: int) -> dict:
        """Calculate cost of false positives.

        Args:
            fp: Number of false positives.

        Returns:
            Dictionary with FP cost breakdown.
        """
        return {
            "n_false_positives": fp,
            "cost_per_fp": self.fp_cost,
            "total_fp_cost": round(fp * self.fp_cost, 2),
        }

    def optimal_threshold(self, y_true, y_proba,
                          thresholds: np.ndarray = None) -> dict:
        """Find classification threshold minimizing total cost.

        Args:
            y_true: Ground truth labels.
            y_proba: Fraud probability scores.
            thresholds: Array of thresholds to evaluate.

        Returns:
            Dictionary with optimal threshold and costs.
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.01)

        costs = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            result = self.analyze(y_true, y_pred)
            total_cost = (result["fraud_missed_value"]
                          + result["fp_cost_total"]
                          + result["investigation_cost_total"])
            costs.append(total_cost)

        min_idx = np.argmin(costs)
        return {
            "optimal_threshold": round(float(thresholds[min_idx]), 2),
            "min_total_cost": round(float(costs[min_idx]), 2),
            "all_thresholds": thresholds.tolist(),
            "all_costs": [round(c, 2) for c in costs],
        }

    def plot_cost_curve(self, y_true, y_proba, save_path: str = None):
        """Plot total cost vs classification threshold.

        Args:
            y_true: Ground truth labels.
            y_proba: Fraud probability scores.
            save_path: Save path for the figure.
        """
        result = self.optimal_threshold(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(result["all_thresholds"], result["all_costs"],
                linewidth=2, color="#3498db")
        ax.axvline(x=result["optimal_threshold"], color="#e74c3c",
                   linestyle="--", label=f"Seuil optimal: {result['optimal_threshold']}")
        ax.set_xlabel("Seuil de Classification")
        ax.set_ylabel("Coût Total (€)")
        ax.set_title("Analyse Coût-Bénéfice: Coût Total vs Seuil")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_figure(fig, save_path or "comparison/cost_curve")
        return fig
