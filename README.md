# Détection Intelligente des Transactions Frauduleuses

Système de détection de fraude basé sur un **ensemble de stacking** avec **intelligence artificielle explicable** (SHAP + LIME).

## Mémoire de Master — 2025/2026

### Architecture

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

### Démarrage immédiat — le modèle est inclus ✅

Le modèle entraîné est **versionné dans le dépôt** (`models/stacking_ensemble.pkl`
+ `models/scaler.pkl`) : l'application fonctionne **dès le clone**, sans
télécharger le jeu de données ni ré-entraîner.

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows : .\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt # dépendances allégées (Python 3.10–3.12)
streamlit run dashboard/app.py       # → http://localhost:8501  (prédictions réelles)
```

### Alternative (script interactif)

```bash
./install.sh   # installe Python venv + dépendances complètes
./run.sh       # menu interactif : dashboard, API, tests, notebooks
```

> `requirements-demo.txt` suffit pour la **démo** (dashboard + API). Pour
> ré-entraîner ou ouvrir les notebooks, utiliser `requirements.txt` complet.

### Guide pratique pour le tuteur

Un guide de lecture et d'exécution plus détaillé est disponible ici :

[`ANNEXE_TECHNIQUE_GUIDE_CODE.md`](ANNEXE_TECHNIQUE_GUIDE_CODE.md)

Il explique en français le rôle de chaque module, l'ordre conseillé pour lire le code, les commandes pour lancer les tests, l'API et le dashboard, ainsi que les limites actuelles du prototype académique.

### Datasets

1. **Kaggle Credit Card**: [Télécharger](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → `data/raw/creditcard.csv`
2. **PaySim**: [Télécharger](https://www.kaggle.com/datasets/ealaxi/paysim1) → `data/raw/paysim.csv`

### Entraîner le modèle (génère le modèle réel)

```bash
python scripts/train_model.py        # ~5-8 min
```

Produit `models/stacking_ensemble.pkl`, `models/scaler.pkl`, les résultats réels
(`models/*.csv`) et des exemples de démonstration. Le dashboard et l'API chargent
automatiquement ce modèle ; sans lui, ils basculent en mode démonstration
(clairement signalé à l'écran).

### Utilisation

**Notebooks** (dans l'ordre):
```bash
cd notebooks/
jupyter notebook 01_EDA_creditcard.ipynb
```

**API**:
```bash
python -m src.api.app
# POST http://localhost:5000/predict
```

**Dashboard**:
```bash
streamlit run dashboard/app.py
```

**Tests**:
```bash
pytest tests/ -v
```

### Structure du Projet

```
├── src/                    # Code source
│   ├── data/               # Chargement, prétraitement, rééchantillonnage
│   ├── models/             # 8 modèles ML (base, ensemble, deep, stacking)
│   ├── training/           # Entraînement, optimisation, validation croisée
│   ├── evaluation/         # Métriques, visualisation, comparaison, coût
│   ├── explainability/     # SHAP, LIME, importance des features
│   └── api/                # API Flask REST
├── dashboard/              # Interface Streamlit (6 pages)
├── scripts/                # train_model.py (entraînement) + download_data.py
├── notebooks/              # 11 notebooks d'analyse
├── tests/                  # Suite de tests pytest (40 tests)
├── models/                 # Résultats (.csv) + modèles entraînés (.pkl, non versionnés)
├── reports/figures/        # Figures générées (EDA, modèles, XAI)
└── ANNEXE_TECHNIQUE_GUIDE_CODE.md  # Guide de lecture du code
```

### Résultats

Chiffres **réels**, mesurés sur le jeu de test (jamais vu), générés par
`python scripts/train_model.py`. Méthodologie : découpage train/validation/test
disjoints, rééquilibrage SMOTE sur l'entraînement, seuil de décision optimisé sur
la validation. Le **stacking obtient la meilleure AUC-ROC**.

| Modèle | AUC-ROC | AUPRC | F1-Score | Précision | Rappel | Temps (ms) |
|---|---|---|---|---|---|---|
| Rég. Logistique | 0.9747 | 0.7245 | 0.7885 | 0.7455 | 0.8367 | 0.05 |
| Arbre de Décision | 0.8538 | 0.5026 | 0.7222 | 0.6610 | 0.7959 | 0.04 |
| Random Forest | 0.9849 | 0.8781 | 0.8442 | 0.8317 | 0.8571 | 14.5 |
| XGBoost | 0.9811 | 0.8785 | 0.8817 | 0.9318 | 0.8367 | 0.22 |
| LightGBM | 0.9812 | 0.8765 | 0.8571 | 0.9740 | 0.7653 | 0.29 |
| MLP | 0.9752 | 0.8458 | 0.7940 | 0.7822 | 0.8061 | 0.06 |
| Auto-encodeur | 0.9616 | 0.2170 | 0.3063 | 0.2207 | 0.5000 | 0.06 |
| **Stacking (proposé)** | **0.9863** | 0.8435 | 0.8216 | 0.8736 | 0.7755 | 15.2 |

> Le stacking maximise l'AUC-ROC (0,9863), la métrique de discrimination globale.
> Les modèles de base restent compétitifs sur d'autres métriques (p. ex. XGBoost
> sur le F1, Random Forest sur le rappel), ce qui est attendu.

### Technologies

Python 3.10+ | scikit-learn | XGBoost | LightGBM | TensorFlow | SHAP | LIME | Flask | Streamlit | Optuna
