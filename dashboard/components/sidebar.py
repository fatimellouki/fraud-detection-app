"""Sidebar navigation component."""

import streamlit as st


def render_sidebar():
    """Render the sidebar with navigation and info."""
    with st.sidebar:
        st.title("🔍 Fraud Detection")
        st.markdown("---")
        st.markdown("""
        **Pages:**
        - 📊 Vue d'ensemble
        - 🔎 Vérificateur
        - 📁 Analyse par lot
        - 📈 Comparaison
        - 🧠 Explicabilité
        - ℹ️ À propos
        """)
        st.markdown("---")
        st.caption("Stacking Ensemble + XAI")
