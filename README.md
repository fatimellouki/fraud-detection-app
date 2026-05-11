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

### Démarrage rapide (2 commandes)

```bash
./install.sh   # installe Python venv, dépendances, datasets
./run.sh       # menu interactif : dashboard, API, tests, notebooks
```

> Si les datasets Kaggle ne se téléchargent pas automatiquement, `install.sh`
> affiche les liens à utiliser pour les placer manuellement dans `data/raw/`.

### Installation manuelle (alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Guide pratique pour le tuteur

Un guide de lecture et d'exécution plus détaillé est disponible ici :

[`ANNEXE_TECHNIQUE_GUIDE_CODE.md`](ANNEXE_TECHNIQUE_GUIDE_CODE.md)

Il explique en français le rôle de chaque module, l'ordre conseillé pour lire le code, les commandes pour lancer les tests, l'API et le dashboard, ainsi que les limites actuelles du prototype académique.

### Datasets

1. **Kaggle Credit Card**: [Télécharger](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → `data/raw/creditcard.csv`
2. **PaySim**: [Télécharger](https://www.kaggle.com/datasets/ealaxi/paysim1) → `data/raw/paysim.csv`

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
├── notebooks/              # 11 notebooks d'analyse
├── rapport/                # Mémoire (~85 pages, français)
│   ├── chapitres/          # 9 chapitres + pages préliminaires
│   ├── diagrams/           # 10 diagrammes Mermaid
│   └── bibliographie/      # 45 références IEEE
├── tests/                  # Suite de tests pytest
└── reports/figures/        # Figures générées (EDA, modèles, XAI)
```

### Résultats

| Modèle | AUC-ROC | F1-Score | Temps (ms) |
|---|---|---|---|
| Rég. Logistique | 0.974 | 0.72 | 0.8 |
| Random Forest | 0.976 | 0.85 | 2.1 |
| XGBoost | 0.981 | 0.88 | 1.5 |
| LightGBM | 0.979 | 0.86 | 1.2 |
| **Stacking (proposé)** | **0.987** | **0.90** | 5.8 |

### Technologies

Python 3.10+ | scikit-learn | XGBoost | LightGBM | TensorFlow | SHAP | LIME | Flask | Streamlit | Optuna
