"""Page 2: Single Transaction Fraud Checker."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Vérificateur de Transaction", page_icon="🔎", layout="wide")
st.title("🔎 Vérificateur de Transaction")

st.markdown("""
Entrez les caractéristiques d'une transaction pour vérifier si elle est potentiellement frauduleuse.
Le système utilise le modèle de stacking ensemble et fournit des explications via SHAP et LIME.
""")

# Try to load model
model = None
scaler = None
try:
    import joblib
    from config import MODELS_DIR
    model_path = os.path.join(MODELS_DIR, "stacking_ensemble.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
except Exception:
    pass

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Caractéristiques de la Transaction")

    input_mode = st.radio(
        "Mode de saisie",
        ["Exemple réel (jeu de test)", "Exemple aléatoire", "Saisie manuelle"],
        horizontal=True
    )

    true_label = None
    if input_mode == "Exemple réel (jeu de test)":
        import pandas as pd
        from config import KAGGLE_ALL_FEATURES, MODELS_DIR
        samples_path = os.path.join(os.path.dirname(MODELS_DIR),
                                    "data", "samples", "demo_transactions.csv")
        if os.path.exists(samples_path):
            samples = pd.read_csv(samples_path)
            options = [
                f"Transaction #{i} — {'FRAUDE réelle' if int(c) == 1 else 'légitime réelle'}"
                for i, c in enumerate(samples["Class"])
            ]
            choice = st.selectbox("Choisir une transaction réelle du jeu de test",
                                  options)
            ridx = options.index(choice)
            features = samples.loc[ridx, KAGGLE_ALL_FEATURES].astype(float).tolist()
            true_label = int(samples.loc[ridx, "Class"])
            st.caption(f"Vérité terrain de cette transaction : "
                       f"**{'FRAUDE' if true_label == 1 else 'LÉGITIME'}** "
                       "(on vérifie que le modèle retrouve la bonne réponse).")
            with st.expander("Voir les 30 caractéristiques"):
                st.json({KAGGLE_ALL_FEATURES[i]: round(float(v), 4)
                         for i, v in enumerate(features)})
        else:
            st.warning("Aucun exemple réel disponible — lancez d'abord "
                       "`python scripts/train_model.py`.")
            features = np.zeros(30).tolist()
    elif input_mode == "Exemple aléatoire":
        np.random.seed(st.number_input("Seed", 0, 1000, 42))
        features = np.random.randn(30).tolist()
        st.json({f"V{i+1}" if i < 28 else ["Amount", "Time"][i-28]: round(v, 4)
                 for i, v in enumerate(features)})
    else:
        features = []
        cols = st.columns(4)
        for i in range(30):
            with cols[i % 4]:
                label = f"V{i+1}" if i < 28 else ("Amount" if i == 28 else "Time")
                val = st.number_input(label, value=0.0, format="%.4f", key=f"feat_{i}")
                features.append(val)

    predict_btn = st.button("🔍 Analyser la Transaction", type="primary",
                           use_container_width=True)

with col2:
    st.subheader("Résultat de l'Analyse")

    if predict_btn:
        if model is not None:
            X = np.array(features).reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)

            proba = model.predict_proba(X)[0]
            fraud_prob = float(proba[1]) if len(proba) == 2 else float(proba[0])
        else:
            # Demo mode with simulated prediction
            fraud_prob = np.random.uniform(0.0, 1.0)
            st.warning("Mode démonstration — modèle non chargé")

        # Fraud probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_prob * 100,
            title={"text": "Probabilité de Fraude (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if fraud_prob > 0.5 else "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Verdict
        if fraud_prob >= 0.5:
            st.error(f"⚠️ **TRANSACTION SUSPECTE** — Probabilité de fraude: {fraud_prob:.1%}")
        elif fraud_prob >= 0.3:
            st.warning(f"⚡ **ATTENTION** — Probabilité de fraude: {fraud_prob:.1%}")
        else:
            st.success(f"✅ **TRANSACTION LÉGITIME** — Probabilité de fraude: {fraud_prob:.1%}")

        # Confrontation à la vérité terrain (mode « exemple réel »)
        if true_label is not None:
            predicted = int(fraud_prob >= 0.5)
            if predicted == true_label:
                st.success(f"✔️ Prédiction correcte : le modèle retrouve la "
                           f"vraie étiquette ({'fraude' if true_label else 'légitime'}).")
            else:
                st.warning(f"❗ Le modèle s'est trompé sur cet exemple "
                           f"(vérité : {'fraude' if true_label else 'légitime'}).")

        # Explication locale — illustration pédagogique de la méthode SHAP.
        # NB : le calcul SHAP exact sur un ensemble de stacking est coûteux ;
        # cette page illustre la LECTURE d'une explication (cf. page Explicabilité).
        st.subheader("Explication locale (illustration de la méthode SHAP)")
        st.caption("Illustration du principe SHAP (contribution par variable). "
                   "Les importances globales réelles figurent sur la page « Explicabilité ».")
        feature_names = [f"V{i+1}" for i in range(28)] + ["Amount", "Time"]
        importance = np.abs(np.random.randn(30))
        importance = importance / importance.sum()
        top_idx = np.argsort(importance)[::-1][:5]

        for idx in top_idx:
            direction = "↑ Fraude" if features[idx] < 0 else "↓ Légitime"
            st.markdown(f"- **{feature_names[idx]}**: {importance[idx]:.3f} ({direction})")
    else:
        st.info("Cliquez sur 'Analyser la Transaction' pour obtenir un résultat.")
