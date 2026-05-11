"""Page 6: About the Project."""

import streamlit as st

st.set_page_config(page_title="À Propos", page_icon="ℹ️", layout="wide")
st.title("ℹ️ À Propos du Projet")

st.markdown("""
## Détection Intelligente des Transactions Frauduleuses dans les Systèmes de Paiement Électroniques

**Mémoire de fin d'études — Master**

### Objectif
Concevoir un système intelligent de détection de fraude basé sur un ensemble
de stacking avec intelligence artificielle explicable (XAI), capable de:

1. **Détecter** les transactions frauduleuses avec une haute précision (AUC-ROC > 0.98)
2. **Expliquer** chaque décision via SHAP et LIME
3. **Déployer** le système sous forme d'application web interactive

### Méthodologie
- **CRISP-DM** (Cross-Industry Standard Process for Data Mining)
- Étude comparative de 8 modèles de machine learning
- Architecture de stacking à deux niveaux
- Explicabilité duale: SHAP (global) + LIME (local)

### Technologies Utilisées
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Machine Learning**
    - Python 3.10+
    - scikit-learn
    - XGBoost
    - LightGBM
    - TensorFlow/Keras
    """)

with col2:
    st.markdown("""
    **Explicabilité**
    - SHAP
    - LIME
    - Permutation Importance

    **Données**
    - Kaggle Credit Card (284K)
    - PaySim (6.3M)
    """)

with col3:
    st.markdown("""
    **Application**
    - Flask (API REST)
    - Streamlit (Dashboard)
    - Plotly (Visualisations)

    **Optimisation**
    - Optuna
    - imbalanced-learn (SMOTE)
    """)

st.markdown("---")

st.markdown("""
### Résultats Principaux

| Aspect | Résultat |
|---|---|
| Meilleur modèle | Stacking Ensemble (RF + XGBoost + LightGBM) |
| AUC-ROC | 0.987 |
| Amélioration vs meilleur individuel | +0.006 (vs XGBoost seul) |
| Meilleure technique d'équilibrage | SMOTE (rappel +10 points) |
| Features les plus importantes | V14, V4, V12, V10, V17 |
| Temps d'inférence | ~5.8 ms par transaction |

### Architecture du Système

```
Niveau 0 (Base):     Random Forest ─┐
                     XGBoost ────────┼── Prédictions de base
                     LightGBM ──────┘
                                     │
Niveau 1 (Méta):     XGBoost (depth=3) ── Prédiction finale
                                     │
Explicabilité:        SHAP + LIME ── Explications
                                     │
Déploiement:          Flask API + Streamlit Dashboard
```
""")

st.markdown("---")
st.caption("Promotion 2024/2025")
