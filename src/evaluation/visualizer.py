"""Publication-quality visualizations for the mémoire report."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

from config import FIGURE_DPI, FIGURE_SIZE_SINGLE, FIGURE_SIZE_GRID, FIGURES_DIR

# Style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": FIGURE_DPI,
})

PALETTE = sns.color_palette("husl", 8)
MODEL_COLORS = {}


def _assign_colors(model_names):
    """Assign consistent colors to model names."""
    global MODEL_COLORS
    for i, name in enumerate(model_names):
        if name not in MODEL_COLORS:
            MODEL_COLORS[name] = PALETTE[i % len(PALETTE)]


def plot_roc_curves(models_dict: dict, save_path: str = None):
    """Plot overlaid ROC curves for all models.

    Args:
        models_dict: {model_name: (y_true, y_proba)} dictionary.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
    _assign_colors(list(models_dict.keys()))

    for name, (y_true, y_proba) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})",
                color=MODEL_COLORS.get(name), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aléatoire")
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.set_title("Courbes ROC — Comparaison des Modèles")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, save_path or "models/roc_curves_all")
    return fig


def plot_pr_curves(models_dict: dict, save_path: str = None):
    """Plot overlaid Precision-Recall curves for all models.

    Args:
        models_dict: {model_name: (y_true, y_proba)} dictionary.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)
    _assign_colors(list(models_dict.keys()))

    for name, (y_true, y_proba) in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = np.trapz(precision, recall) * -1 if recall[0] > recall[-1] else np.trapz(precision, recall)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})",
                color=MODEL_COLORS.get(name), linewidth=2)

    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbes Précision-Rappel — Comparaison des Modèles")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_figure(fig, save_path or "models/pr_curves_all")
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name: str,
                          save_path: str = None):
    """Plot annotated heatmap confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Name for the title.
        save_path: If provided, save figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Légitime", "Fraude"],
        yticklabels=["Légitime", "Fraude"],
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")
    ax.set_title(f"Matrice de Confusion — {model_name}")
    plt.tight_layout()

    save_figure(fig, save_path or f"models/cm_{model_name.lower().replace(' ', '_')}")
    return fig


def plot_confusion_matrices_grid(models_dict: dict, save_path: str = None):
    """Plot a grid of confusion matrices for all models.

    Args:
        models_dict: {model_name: (y_true, y_pred)} dictionary.
        save_path: If provided, save figure to this path.
    """
    n = len(models_dict)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=FIGURE_SIZE_GRID)
    axes = axes.flatten()

    for idx, (name, (y_true, y_pred)) in enumerate(models_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Lég.", "Fr."],
            yticklabels=["Lég.", "Fr."],
            ax=axes[idx], cbar=False
        )
        axes[idx].set_title(name, fontsize=11)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Matrices de Confusion — Tous les Modèles", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path or "models/confusion_matrices_grid")
    return fig


def plot_class_distribution(y, title: str = "Distribution des Classes",
                            save_path: str = None):
    """Plot bar chart of class balance.

    Args:
        y: Target vector.
        title: Plot title.
        save_path: Save path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts = np.bincount(np.asarray(y).astype(int))
    labels = ["Légitime", "Fraude"]
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(labels, counts, color=colors, edgecolor="black", alpha=0.8)
    axes[0].set_ylabel("Nombre de Transactions")
    axes[0].set_title(title)
    for i, v in enumerate(counts):
        axes[0].text(i, v + max(counts)*0.01, f"{v:,}", ha="center", fontsize=11)

    # Pie chart
    axes[1].pie(counts, labels=labels, colors=colors, autopct="%1.3f%%",
                startangle=90, explode=(0, 0.1))
    axes[1].set_title("Proportion des Classes")

    plt.tight_layout()
    save_figure(fig, save_path or "eda/class_distribution")
    return fig


def plot_feature_correlation(X, save_path: str = None):
    """Plot correlation heatmap.

    Args:
        X: Feature DataFrame.
        save_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                ax=ax, square=True, linewidths=0.5, fmt=".2f")
    ax.set_title("Matrice de Corrélation des Features")
    plt.tight_layout()
    save_figure(fig, save_path or "eda/correlation_heatmap")
    return fig


def plot_amount_distribution(df, save_path: str = None):
    """Plot histogram of transaction amounts by class.

    Args:
        df: DataFrame with 'Amount' and 'Class' columns.
        save_path: Save path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    target = "Class" if "Class" in df.columns else "isFraud"

    # Linear scale
    axes[0].hist(df[df[target] == 0]["Amount"], bins=50, alpha=0.7,
                 label="Légitime", color="#2ecc71", density=True)
    axes[0].hist(df[df[target] == 1]["Amount"], bins=50, alpha=0.7,
                 label="Fraude", color="#e74c3c", density=True)
    axes[0].set_xlabel("Montant (€)")
    axes[0].set_ylabel("Densité")
    axes[0].set_title("Distribution des Montants")
    axes[0].legend()

    # Log scale
    amount_col = "Amount" if "Amount" in df.columns else "amount"
    log_amounts = np.log1p(df[amount_col])
    axes[1].hist(log_amounts[df[target] == 0], bins=50, alpha=0.7,
                 label="Légitime", color="#2ecc71", density=True)
    axes[1].hist(log_amounts[df[target] == 1], bins=50, alpha=0.7,
                 label="Fraude", color="#e74c3c", density=True)
    axes[1].set_xlabel("log(1 + Montant)")
    axes[1].set_ylabel("Densité")
    axes[1].set_title("Distribution des Montants (échelle log)")
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, save_path or "eda/amount_histogram")
    return fig


def plot_time_distribution(df, save_path: str = None):
    """Plot transaction frequency over time.

    Args:
        df: DataFrame with 'Time' and 'Class' columns.
        save_path: Save path.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    target = "Class" if "Class" in df.columns else "isFraud"
    time_col = "Time" if "Time" in df.columns else "step"

    # Convert to hours
    hours = df[time_col] / 3600

    ax.hist(hours[df[target] == 0], bins=48, alpha=0.6,
            label="Légitime", color="#2ecc71", density=True)
    ax.hist(hours[df[target] == 1], bins=48, alpha=0.6,
            label="Fraude", color="#e74c3c", density=True)
    ax.set_xlabel("Temps (heures)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution Temporelle des Transactions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, save_path or "eda/time_distribution")
    return fig


def plot_model_comparison_bar(results_df, metrics=None, save_path: str = None):
    """Plot grouped bar chart comparing all models on multiple metrics.

    Args:
        results_df: DataFrame with model names as index and metrics as columns.
        metrics: List of metric columns to plot. Default: main metrics.
        save_path: Save path.
    """
    if metrics is None:
        metrics = ["auc_roc", "f1_score", "precision", "recall", "auprc"]

    available = [m for m in metrics if m in results_df.columns]
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(results_df))
    width = 0.8 / len(available)

    for i, metric in enumerate(available):
        offset = (i - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, results_df[metric], width,
               label=metric.replace("_", " ").title(), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des Modèles — Métriques Principales")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, save_path or "comparison/model_comparison_bar")
    return fig


def plot_imbalance_comparison(results_dict: dict, save_path: str = None):
    """Plot comparison of imbalance technique results.

    Args:
        results_dict: {technique_name: metrics_dict} dictionary.
        save_path: Save path.
    """
    import pandas as pd

    df = pd.DataFrame(results_dict).T
    metrics = ["auc_roc", "f1_score", "precision", "recall"]
    available = [m for m in metrics if m in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    df[available].plot(kind="bar", ax=ax, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Impact du Traitement du Déséquilibre des Classes")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    save_figure(fig, save_path or "comparison/imbalance_comparison")
    return fig


def save_figure(fig, name: str, subdir: str = None):
    """Save figure to the reports/figures directory.

    Args:
        fig: Matplotlib figure.
        name: Filename (without extension) or relative path from FIGURES_DIR.
        subdir: Optional subdirectory under FIGURES_DIR.
    """
    if subdir:
        path = os.path.join(FIGURES_DIR, subdir, f"{name}.{plt.rcParams.get('savefig.format', 'png')}")
    else:
        path = os.path.join(FIGURES_DIR, f"{name}.png")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
