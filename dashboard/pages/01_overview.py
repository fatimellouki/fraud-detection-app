"""Page 1: System Overview & Statistics."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "components"))
from results import load_results

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

df_results, _is_real = load_results()
if not _is_real:
    st.caption("⚠️ Modèle non entraîné — valeurs de repli. Lancez "
               "`python scripts/train_model.py` pour des chiffres réels.")

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
