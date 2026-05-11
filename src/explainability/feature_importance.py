"""Additional feature importance methods: permutation importance, PDP."""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from config import FIGURES_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)


def compute_permutation_importance(model, X, y, n_repeats: int = 10,
                                   scoring: str = "roc_auc",
                                   feature_names: list = None,
                                   save_path: str = None) -> dict:
    """Compute and plot permutation importance.

    Args:
        model: Trained model.
        X: Feature matrix.
        y: Target vector.
        n_repeats: Number of permutation repeats.
        scoring: Metric to evaluate.
        feature_names: Feature names.
        save_path: Save path for the plot.

    Returns:
        Dictionary with importance values and ranking.
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=42,
        n_jobs=-1
    )

    importances = result.importances_mean
    std = result.importances_std
    indices = np.argsort(importances)[::-1][:15]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [feature_names[i] for i in indices]
    vals = importances[indices]
    errs = std[indices]

    ax.barh(range(len(names)), vals, xerr=errs, align="center",
            color="#3498db", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"Diminution de {scoring}")
    ax.set_title("Importance par Permutation (Top 15)")
    plt.tight_layout()

    import os
    path = save_path or os.path.join(FIGURES_DIR, "explainability", "permutation_importance.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()

    return {
        "feature_names": [feature_names[i] for i in indices],
        "importances": importances[indices].tolist(),
        "std": std[indices].tolist(),
    }
