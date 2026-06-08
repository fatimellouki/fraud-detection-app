"""Page 4: Interactive Model Comparison."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "components"))
from results import load_results

st.set_page_config(page_title="Comparaison des Modèles", page_icon="📈", layout="wide")
st.title("📈 Comparaison des Modèles")

# Résultats réels (models/results_comparison.csv) — source unique de vérité.
df, _is_real = load_results()
if not _is_real:
    st.caption("⚠️ Modèle non entraîné — valeurs de repli. Lancez "
               "`python scripts/train_model.py` pour des chiffres réels.")

# Metrics table
st.subheader("Tableau Comparatif")
st.dataframe(
    df.style
    .highlight_max(subset=["AUC-ROC", "AUPRC", "F1-Score", "Précision", "Rappel"], color="#90EE90")
    .highlight_min(subset=["Temps (ms)"], color="#90EE90")
    .format({col: "{:.4f}" for col in df.columns if col not in ["Modèle", "Temps (ms)"]})
    .format({"Temps (ms)": "{:.1f}"}),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Interactive metric selection
col1, col2 = st.columns([1, 2])

with col1:
    selected_metrics = st.multiselect(
        "Métriques à afficher",
        ["AUC-ROC", "AUPRC", "F1-Score", "Précision", "Rappel"],
        default=["AUC-ROC", "F1-Score", "Rappel"]
    )

with col2:
    if selected_metrics:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=df["Modèle"],
                y=df[metric],
                marker_color=colors[i % len(colors)]
            ))
        fig.update_layout(
            title="Comparaison des Modèles",
            barmode="group",
            yaxis_range=[0, 1.05],
            xaxis_tickangle=-45,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ROC curves (illustratives — reconstruites à partir de l'AUC réelle)
st.subheader("Courbes ROC")
st.caption("Courbes illustratives reconstruites à partir de l'AUC mesurée de "
           "chaque modèle (les valeurs d'AUC du tableau, elles, sont réelles).")
fig_roc = go.Figure()

for i, model_name in enumerate(df["Modèle"]):
    auc_val = df.loc[i, "AUC-ROC"]
    # Generate smooth ROC-like curve
    fpr = np.sort(np.concatenate([[0], np.random.beta(0.5, auc_val * 10, 50), [1]]))
    tpr = np.sort(np.concatenate([[0], np.random.beta(auc_val * 10, 0.5, 50), [1]]))
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC={auc_val:.3f})",
    ))

fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Aléatoire",
    line=dict(dash="dash", color="gray")
))
fig_roc.update_layout(
    title="Courbes ROC — Tous les Modèles",
    xaxis_title="Taux de Faux Positifs (FPR)",
    yaxis_title="Taux de Vrais Positifs (TPR)",
    height=500
)
st.plotly_chart(fig_roc, use_container_width=True)

# Radar chart
st.subheader("Profil de Performance (Radar)")

selected_model = st.selectbox("Sélectionner un modèle", df["Modèle"])
idx = df[df["Modèle"] == selected_model].index[0]

categories = ["AUC-ROC", "AUPRC", "F1-Score", "Précision", "Rappel"]
values = [df.loc[idx, c] for c in categories]
values.append(values[0])  # Close the polygon

fig_radar = go.Figure(data=go.Scatterpolar(
    r=values,
    theta=categories + [categories[0]],
    fill="toself",
    name=selected_model
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title=f"Profil de Performance — {selected_model}",
    height=450
)
st.plotly_chart(fig_radar, use_container_width=True)
