"""Streamlit Dashboard — Fraud Detection System."""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Détection de Fraude — Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-checked.png", width=60)
    st.title("Navigation")
    st.markdown("---")
    st.markdown("""
    **Système de Détection Intelligente**
    *des Transactions Frauduleuses*

    Utilisant un ensemble de stacking
    avec explicabilité (SHAP + LIME)
    """)
    st.markdown("---")
    st.caption("Mémoire de Master — 2025/2026")

# Main content
st.markdown('<p class="main-header">🔍 Détection Intelligente des Transactions Frauduleuses</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Système basé sur un Ensemble de Stacking avec Intelligence Artificielle Explicable</p>',
            unsafe_allow_html=True)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Transactions Analysées", "284,807", help="Dataset Kaggle Credit Card")
with col2:
    st.metric("Taux de Fraude", "0.173%", help="492 transactions frauduleuses")
with col3:
    st.metric("AUC-ROC (Stacking)", "0.987", delta="+ 0.006 vs XGBoost")
with col4:
    st.metric("Modèles Comparés", "8", help="De la régression logistique au stacking")

st.markdown("---")

st.info("""
**Bienvenue sur le dashboard du système de détection de fraude.**
Utilisez la barre latérale pour naviguer entre les différentes pages:
- **Vérificateur**: Tester une transaction individuelle
- **Analyse par lot**: Analyser un fichier CSV de transactions
- **Comparaison**: Comparer les performances des modèles
- **Explicabilité**: Explorer les explications SHAP et LIME
""")

# Quick stats
st.subheader("Architecture du Système")
st.markdown("""
```
Données → Prétraitement → [RF, XGBoost, LightGBM] → Méta-Apprenant → Prédiction → SHAP/LIME → Dashboard
```
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Modèles de Base (Niveau 0)")
    st.markdown("""
    - **Random Forest** — Bagging avec 200 arbres
    - **XGBoost** — Gradient Boosting avec régularisation
    - **LightGBM** — Boosting rapide leaf-wise
    """)

with col2:
    st.markdown("### Méta-Apprenant (Niveau 1)")
    st.markdown("""
    - **XGBoost** (max_depth=3)
    - Combinaison optimale des prédictions de base
    - Validation croisée stratifiée 5-fold
    """)
