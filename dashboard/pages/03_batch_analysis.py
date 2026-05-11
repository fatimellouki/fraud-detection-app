"""Page 3: Batch CSV Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Analyse par Lot", page_icon="📁", layout="wide")
st.title("📁 Analyse par Lot")

st.markdown("""
Téléchargez un fichier CSV contenant des transactions pour une analyse en masse.
Le système attribuera un score de fraude à chaque transaction.
""")

uploaded_file = st.file_uploader(
    "Choisir un fichier CSV",
    type=["csv"],
    help="Le fichier doit contenir les 30 features (V1-V28, Amount, Time)"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Fichier chargé: {df.shape[0]:,} transactions, {df.shape[1]} colonnes")

    st.subheader("Aperçu des Données")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("🔍 Analyser toutes les transactions", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Try to load model, fallback to demo mode
            try:
                import joblib
                from config import MODELS_DIR
                model = joblib.load(os.path.join(MODELS_DIR, "stacking_ensemble.pkl"))
                feature_cols = [c for c in df.columns if c not in ["Class", "isFraud"]]
                probas = model.predict_proba(df[feature_cols])[:, 1]
            except Exception:
                probas = np.random.beta(1, 20, size=len(df))
                st.warning("Mode démonstration — scores simulés")

            df["fraud_score"] = probas
            df["is_flagged"] = (probas >= 0.5).astype(int)

        # Results
        st.subheader("Résultats de l'Analyse")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analysées", f"{len(df):,}")
        with col2:
            flagged = df["is_flagged"].sum()
            st.metric("Transactions Suspectes", f"{flagged:,}",
                      delta=f"{flagged/len(df)*100:.2f}%")
        with col3:
            st.metric("Score Moyen de Fraude", f"{df['fraud_score'].mean():.4f}")

        # Score distribution
        fig = px.histogram(
            df, x="fraud_score", nbins=50,
            title="Distribution des Scores de Fraude",
            color_discrete_sequence=["#3498db"]
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Seuil (0.5)")
        st.plotly_chart(fig, use_container_width=True)

        # Flagged transactions
        st.subheader("Transactions Suspectes (Score ≥ 0.5)")
        flagged_df = df[df["is_flagged"] == 1].sort_values("fraud_score", ascending=False)
        st.dataframe(flagged_df, use_container_width=True)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les Résultats (CSV)",
            data=csv,
            file_name="fraud_analysis_results.csv",
            mime="text/csv"
        )
else:
    st.info("Téléchargez un fichier CSV pour commencer l'analyse.")

    # Demo data
    st.subheader("Format Attendu")
    demo = pd.DataFrame(
        np.random.randn(5, 30),
        columns=[f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    )
    st.dataframe(demo, use_container_width=True)
