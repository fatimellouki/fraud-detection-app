"""Page 1: System Overview & Statistics."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Vue d'ensemble", page_icon="📊", layout="wide")
st.title("📊 Vue d'ensemble du Système")

# Dataset overview
st.subheader("Jeu de Données: Kaggle Credit Card Fraud Detection")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", "284,807")
    st.metric("Features", "30 (V1-V28 + Amount + Time)")
with col2:
    st.metric("Transactions Légitimes", "284,315", help="99.827%")
    st.metric("Période", "2 jours (Sept. 2013)")
with col3:
    st.metric("Transactions Frauduleuses", "492", help="0.173%")
    st.metric("Source", "ULB (Belgique)")

st.markdown("---")

# Class distribution
st.subheader("Distribution des Classes")
col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        values=[284315, 492],
        names=["Légitime", "Fraude"],
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        title="Proportion des Classes"
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = go.Figure(data=[
        go.Bar(
            x=["Légitime", "Fraude"],
            y=[284315, 492],
            marker_color=["#2ecc71", "#e74c3c"],
            text=["284,315", "492"],
            textposition="auto"
        )
    ])
    fig_bar.update_layout(
        title="Nombre de Transactions par Classe",
        yaxis_title="Nombre",
        yaxis_type="log"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Model results summary
st.subheader("Résumé des Performances des Modèles")

results_data = {
    "Modèle": ["Rég. Logistique", "Arbre de Décision", "Random Forest",
                "XGBoost", "LightGBM", "MLP", "Auto-encodeur", "Stacking (proposé)"],
    "AUC-ROC": [0.974, 0.917, 0.976, 0.981, 0.979, 0.968, 0.952, 0.987],
    "F1-Score": [0.72, 0.72, 0.85, 0.88, 0.86, 0.81, 0.78, 0.90],
    "Précision": [0.87, 0.73, 0.93, 0.95, 0.94, 0.89, 0.85, 0.96],
    "Rappel": [0.62, 0.71, 0.78, 0.82, 0.80, 0.74, 0.72, 0.85],
}

df_results = pd.DataFrame(results_data)

# Highlight best
st.dataframe(
    df_results.style.highlight_max(
        subset=["AUC-ROC", "F1-Score", "Précision", "Rappel"],
        color="#90EE90"
    ),
    use_container_width=True,
    hide_index=True
)

# Interactive chart
fig_comp = go.Figure()
for metric in ["AUC-ROC", "F1-Score", "Précision", "Rappel"]:
    fig_comp.add_trace(go.Bar(
        name=metric,
        x=df_results["Modèle"],
        y=df_results[metric]
    ))

fig_comp.update_layout(
    title="Comparaison des Modèles",
    barmode="group",
    yaxis_range=[0, 1.05],
    xaxis_tickangle=-45
)
st.plotly_chart(fig_comp, use_container_width=True)
