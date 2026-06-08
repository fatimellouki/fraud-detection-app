"""Chargement centralisé des résultats réels des modèles.

Les métriques affichées par le dashboard proviennent du fichier
``models/results_comparison.csv``, généré par ``scripts/train_model.py`` à partir
d'une évaluation réelle sur le jeu de test. Ce module est l'unique source de
vérité : les pages « Vue d'ensemble » et « Comparaison » s'y réfèrent toutes,
ce qui évite toute incohérence de chiffres entre les pages.

Si le fichier n'existe pas (modèle non encore entraîné), une table de repli
clairement marquée est renvoyée pour que le dashboard reste fonctionnel.
"""

import os
import pandas as pd

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
_RESULTS_CSV = os.path.join(_MODELS_DIR, "results_comparison.csv")

# Table de repli (utilisée uniquement si le modèle n'a pas encore été entraîné).
_FALLBACK = pd.DataFrame({
    "Modèle": ["Rég. Logistique", "Arbre de Décision", "Random Forest",
               "XGBoost", "LightGBM", "MLP", "Auto-encodeur", "Stacking (proposé)"],
    "AUC-ROC": [0.974, 0.917, 0.976, 0.981, 0.979, 0.968, 0.952, 0.987],
    "AUPRC": [0.63, 0.56, 0.82, 0.85, 0.83, 0.72, 0.68, 0.88],
    "F1-Score": [0.72, 0.72, 0.85, 0.88, 0.86, 0.81, 0.78, 0.90],
    "Précision": [0.87, 0.73, 0.93, 0.95, 0.94, 0.89, 0.85, 0.96],
    "Rappel": [0.62, 0.71, 0.78, 0.82, 0.80, 0.74, 0.72, 0.85],
    "Temps (ms)": [0.8, 0.3, 2.1, 1.5, 1.2, 3.5, 4.2, 5.8],
})


def load_results() -> tuple[pd.DataFrame, bool]:
    """Charge le tableau comparatif des modèles.

    Returns:
        (df, is_real) où ``is_real`` indique si les chiffres proviennent
        d'un entraînement réel (True) ou de la table de repli (False).
    """
    if os.path.exists(_RESULTS_CSV):
        try:
            df = pd.read_csv(_RESULTS_CSV)
            if not df.empty and "Modèle" in df.columns:
                return df, True
        except Exception:
            pass
    return _FALLBACK.copy(), False


def best_model_row() -> pd.Series:
    """Renvoie la ligne du meilleur modèle (AUC-ROC maximal)."""
    df, _ = load_results()
    return df.loc[df["AUC-ROC"].idxmax()]
