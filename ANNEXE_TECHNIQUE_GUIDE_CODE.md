# Annexe technique : Guide pratique de lecture et d'exécution du code

## 1. But de ce document

Ce document sert de guide pratique pour comprendre le code du projet. Il est destiné à l'encadrant, au tuteur ou à un membre du jury qui souhaite ouvrir le dépôt et comprendre rapidement ce que chaque partie du code réalise.

Le projet est un prototype académique de détection de fraude dans les paiements électroniques. La partie théorique du mémoire explique les notions : fraude, déséquilibre des classes, apprentissage automatique, stacking, SHAP, LIME et déploiement. Le code montre comment ces notions sont traduites en modules Python.

L'idée générale est simple :

```text
On donne au système des transactions.
Le système apprend à distinguer les transactions normales des transactions frauduleuses.
Ensuite, il peut recevoir une nouvelle transaction et estimer si elle est suspecte.
```

## 2. Explication simple du fonctionnement général

Le projet peut être compris comme une chaîne de traitement en dix étapes.

### Étape 1 : Charger les données

Le système a besoin d'un tableau de transactions. Chaque ligne représente une transaction. Certaines transactions sont normales, d'autres sont frauduleuses.

Exemple simplifié :

```text
Transaction 1 -> normale
Transaction 2 -> normale
Transaction 3 -> fraude
Transaction 4 -> normale
```

Les jeux de données prévus sont :

- `creditcard.csv`, issu du jeu Kaggle Credit Card Fraud Detection ;
- `paysim.csv`, issu du jeu PaySim.

Ces fichiers ne sont pas inclus dans le dépôt, car ils sont volumineux. Ils doivent être placés dans `data/raw/`.

### Étape 2 : Nettoyer et préparer les données

Les modèles de machine learning ne travaillent pas correctement si les données sont mal préparées. Le code vérifie donc les valeurs manquantes, transforme certaines variables et normalise les grandeurs numériques.

Par exemple, le montant d'une transaction peut varier fortement. Une transformation logarithmique permet de réduire cet écart et de rendre la variable plus facile à exploiter par les modèles.

Cette étape correspond au module :

```text
src/data/preprocessor.py
```

### Étape 3 : Corriger le déséquilibre entre fraude et non-fraude

Dans la réalité, la fraude est rare. Un jeu de données peut contenir des centaines de milliers de transactions normales et seulement quelques centaines de fraudes.

Le problème est le suivant : si le modèle voit presque uniquement des transactions normales, il risque d'apprendre à répondre toujours "normale". Il aura alors une très bonne exactitude apparente, mais il sera inutile pour détecter la fraude.

Pour éviter cela, le code utilise des méthodes comme SMOTE et ADASYN. Ces méthodes créent des exemples synthétiques de fraudes pendant l'entraînement afin d'aider le modèle à mieux reconnaître les cas rares.

Cette étape correspond au fichier :

```text
src/data/balancer.py
```

### Étape 4 : Entraîner plusieurs modèles

Le projet ne teste pas un seul modèle. Il compare plusieurs approches :

- régression logistique ;
- arbre de décision ;
- Random Forest ;
- XGBoost ;
- LightGBM ;
- CatBoost ;
- réseau de neurones MLP ;
- auto-encodeur.

Chaque modèle apprend à partir des transactions connues. Le but est de savoir quel modèle détecte le mieux les fraudes.

Les modèles sont définis dans :

```text
src/models/
```

### Étape 5 : Combiner les modèles avec le stacking

La contribution principale du projet est le stacking. Au lieu de faire confiance à un seul modèle, le système combine plusieurs modèles forts.

Dans ce projet, les trois modèles de base sont :

```text
Random Forest
XGBoost
LightGBM
```

Chaque modèle donne son avis sur une transaction. Ensuite, un autre modèle, appelé méta-apprenant, apprend à combiner ces avis pour donner la décision finale.

Exemple simplifié :

```text
Random Forest -> 70 % de risque de fraude
XGBoost       -> 85 % de risque de fraude
LightGBM      -> 78 % de risque de fraude

Méta-modèle   -> décision finale : transaction suspecte
```

Le fichier principal de cette partie est :

```text
src/models/stacking_model.py
```

### Étape 6 : Évaluer les résultats

Après l'entraînement, le système est testé sur des transactions qu'il n'a jamais vues. On mesure alors sa performance.

Les métriques utilisées sont adaptées à la fraude :

- `precision` : quand le modèle annonce une fraude, à quelle fréquence a-t-il raison ?
- `recall` : parmi toutes les vraies fraudes, combien sont détectées ?
- `f1_score` : équilibre entre précision et rappel ;
- `auc_roc` : capacité générale du modèle à séparer fraude et non-fraude ;
- `auprc` : métrique très utile quand la fraude est rare ;
- matrice de confusion : nombre de vrais positifs, faux positifs, vrais négatifs et faux négatifs.

Cette partie est dans :

```text
src/evaluation/
```

### Étape 7 : Expliquer les décisions

Dans un contexte bancaire, il ne suffit pas de dire "cette transaction est frauduleuse". Il faut aussi expliquer pourquoi le modèle pense cela.

Le projet utilise deux outils :

- SHAP, pour expliquer globalement et localement l'influence des variables ;
- LIME, pour expliquer une prédiction individuelle.

Exemple d'explication simplifiée :

```text
La transaction est considérée comme suspecte car les variables V14, V4 et V12 ont des valeurs inhabituelles.
```

Cette partie correspond à :

```text
src/explainability/
```

### Étape 8 : Exposer le modèle par une API

Une API permet à une autre application d'envoyer une transaction au modèle et de recevoir une réponse.

Exemple de réponse attendue :

```json
{
  "fraud_probability": 0.87,
  "is_fraud": true
}
```

L'API est développée avec Flask dans :

```text
src/api/
```

### Étape 9 : Afficher les résultats dans un tableau de bord

Le tableau de bord Streamlit permet d'interagir visuellement avec le projet. Il sert à :

- consulter une vue générale du système ;
- tester une transaction individuelle ;
- analyser un fichier CSV de transactions ;
- comparer les modèles ;
- visualiser les explications SHAP et LIME.

Le code du tableau de bord se trouve dans :

```text
dashboard/
```

### Étape 10 : Vérifier le code avec des tests

Le dossier `tests/` contient des tests automatiques. Ils vérifient que les principales briques du projet fonctionnent : chargement, prétraitement, rééquilibrage, modèles, stacking, métriques et API.

Lors du retest local du 9 mai 2026, le résultat obtenu est :

```text
40 passed, 3 warnings in 4.82s
```

Cela signifie que les tests unitaires passent dans un environnement Python 3.11 correctement configuré.

## 3. Rôle des principaux fichiers

Le tableau suivant indique les fichiers les plus importants et leur rôle dans le projet.

| Fichier ou dossier | Rôle pratique |
|---|---|
| `README.md` | Présentation générale du projet, installation et commandes principales. |
| `requirements.txt` | Liste des bibliothèques Python nécessaires. |
| `config.py` | Chemins des dossiers, noms de datasets, paramètres des modèles, constantes globales. |
| `scripts/download_data.py` | Vérifie si les datasets attendus sont présents dans `data/raw/`. |
| `src/data/loader.py` | Charge les fichiers CSV et vérifie la structure des données. |
| `src/data/preprocessor.py` | Nettoie les données, crée des variables dérivées et applique la normalisation. |
| `src/data/splitter.py` | Découpe les données en ensembles train, validation et test avec stratification. |
| `src/data/balancer.py` | Applique SMOTE, ADASYN, sous-échantillonnage et pondération des classes. |
| `src/models/base_models.py` | Définit les modèles simples : régression logistique et arbre de décision. |
| `src/models/ensemble_models.py` | Définit Random Forest, XGBoost, LightGBM et CatBoost. |
| `src/models/deep_models.py` | Définit le MLP et l'auto-encodeur. |
| `src/models/stacking_model.py` | Implémente le modèle principal de stacking. |
| `src/models/model_factory.py` | Permet de créer un modèle à partir de son nom. |
| `src/training/trainer.py` | Entraîne un modèle, mesure le temps et calcule les métriques. |
| `src/training/cross_validator.py` | Réalise une validation croisée stratifiée. |
| `src/training/hypertuner.py` | Optimise certains hyperparamètres avec Optuna. |
| `src/evaluation/metrics.py` | Calcule les métriques d'évaluation. |
| `src/evaluation/visualizer.py` | Génère les courbes ROC, Precision-Recall et matrices de confusion. |
| `src/evaluation/comparator.py` | Compare les résultats de plusieurs modèles. |
| `src/evaluation/cost_analysis.py` | Estime l'impact économique des faux positifs et faux négatifs. |
| `src/explainability/shap_explainer.py` | Produit des explications SHAP. |
| `src/explainability/lime_explainer.py` | Produit des explications LIME. |
| `src/api/app.py` | Crée l'application Flask et charge les artefacts du modèle. |
| `src/api/routes.py` | Définit les endpoints `/health`, `/predict`, `/predict/batch`, `/explain`, `/model/info`. |
| `dashboard/app.py` | Page principale du dashboard Streamlit. |
| `dashboard/pages/` | Pages secondaires du dashboard : vérification, batch, comparaison, explicabilité, etc. |
| `notebooks/` | Expérimentations étape par étape, de l'EDA à l'évaluation finale. |
| `models/final_results.csv` | Résultats finaux présentés pour les modèles. |
| `tests/` | Tests automatisés du code. |

## 4. Guide d'exécution du projet

### 4.1 Préparer l'environnement

Il est recommandé d'utiliser Python 3.10 ou Python 3.11. Python 3.14 est trop récent pour certains projets de machine learning et peut poser des problèmes de compatibilité.

Depuis la racine du projet :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Si l'installation complète est trop lourde à cause de TensorFlow, il est possible de commencer par les dépendances nécessaires aux tests unitaires :

```bash
python -m pip install pytest numpy pandas scikit-learn imbalanced-learn xgboost lightgbm catboost flask flask-cors joblib matplotlib plotly
```

### 4.2 Vérifier la présence des datasets

Lancer :

```bash
python scripts/download_data.py
```

Résultat attendu si les datasets ne sont pas présents :

```text
[MANQUANT] Kaggle Credit Card
[MANQUANT] PaySim
```

Il faut alors télécharger les fichiers depuis Kaggle et les placer ici :

```text
data/raw/creditcard.csv
data/raw/paysim.csv
```

### 4.3 Exécuter les notebooks dans l'ordre

Les notebooks représentent la partie expérimentale du projet. L'ordre conseillé est :

| Ordre | Notebook | Rôle |
|---|---|---|
| 1 | `01_EDA_creditcard.ipynb` | Analyse exploratoire du dataset Kaggle Credit Card. |
| 2 | `02_EDA_paysim.ipynb` | Analyse exploratoire du dataset PaySim. |
| 3 | `03_preprocessing.ipynb` | Prétraitement, création des splits et sauvegarde du scaler. |
| 4 | `04_imbalance_comparison.ipynb` | Comparaison des techniques SMOTE, ADASYN, etc. |
| 5 | `05_baseline_models.ipynb` | Entraînement des modèles simples. |
| 6 | `06_ensemble_models.ipynb` | Entraînement des modèles Random Forest, XGBoost, LightGBM, CatBoost. |
| 7 | `07_deep_learning.ipynb` | Tests avec MLP et auto-encodeur. |
| 8 | `08_stacking_ensemble.ipynb` | Entraînement du modèle de stacking final. |
| 9 | `09_explainability.ipynb` | Génération des explications SHAP et LIME. |
| 10 | `10_final_evaluation.ipynb` | Évaluation finale et comparaison des modèles. |
| 11 | `11_paysim_generalization.ipynb` | Test de généralisation sur PaySim. |

Le notebook le plus important pour le modèle final est :

```text
notebooks/08_stacking_ensemble.ipynb
```

Il doit produire le fichier :

```text
models/stacking_ensemble.pkl
```

Le notebook de prétraitement doit produire :

```text
models/scaler.pkl
```

Ces deux fichiers sont nécessaires pour que l'API et le dashboard fonctionnent avec un vrai modèle.

### 4.4 Lancer les tests

Commande :

```bash
python -m pytest tests -q
```

Résultat attendu dans un environnement correct :

```text
40 passed, 3 warnings
```

Les avertissements observés ne bloquent pas l'exécution :

- la régression logistique peut atteindre la limite de convergence ;
- XGBoost signale que `use_label_encoder` n'est plus utilisé dans les versions récentes.

### 4.5 Lancer l'API Flask

Commande :

```bash
python -m src.api.app
```

L'API démarre sur :

```text
http://localhost:5000
```

Pour tester l'état de l'API :

```bash
curl http://localhost:5000/health
```

Si `models/stacking_ensemble.pkl` est présent, la réponse doit indiquer que le modèle est chargé. Sinon, l'API répond en mode dégradé.

### 4.6 Tester une prédiction par API

Exemple de requête :

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 3600]}'
```

Réponse attendue si le modèle est chargé :

```json
{
  "fraud_probability": 0.1234,
  "is_fraud": false,
  "threshold": 0.5
}
```

La valeur exacte dépend du modèle entraîné.

### 4.7 Lancer le dashboard Streamlit

Commande :

```bash
streamlit run dashboard/app.py
```

Le dashboard permet de naviguer entre plusieurs pages :

- vue d'ensemble ;
- vérificateur de transaction ;
- analyse par lot ;
- comparaison des modèles ;
- explicabilité ;
- à propos.

Si le modèle entraîné n'est pas présent, certaines pages passent en mode démonstration. Dans ce cas, les scores affichés ne doivent pas être interprétés comme des prédictions réelles.

## 5. Résultats attendus

Le fichier `models/final_results.csv` indique les performances suivantes pour le stacking :

```text
AUC-ROC  : 0.9844
AUPRC    : 0.8732
F1-score : 0.8426
Precision: 0.8384
Recall   : 0.8469
```

Ces résultats montrent que le stacking obtient un bon compromis entre détection des fraudes et limitation des fausses alertes.

Cependant, une limite importante doit être signalée : le fichier `models/results_comparison.csv` contient une autre ligne pour le stacking, avec un AUC-ROC de `0.9696`. Cette divergence doit être clarifiée avant la version finale du mémoire. Elle peut venir d'une différence de protocole : avec ou sans SMOTE, version intermédiaire du modèle, ou exécution différente des notebooks.

## 6. Ce que le projet valide

Le projet valide surtout la faisabilité technique de l'approche décrite dans le mémoire.

Il montre que l'on peut :

- charger et préparer des données transactionnelles ;
- traiter le déséquilibre extrême entre fraude et non-fraude ;
- entraîner plusieurs modèles ;
- construire un modèle de stacking ;
- comparer les performances ;
- générer des explications SHAP et LIME ;
- exposer le modèle par API ;
- afficher les résultats dans un dashboard ;
- tester automatiquement les principales briques du code.

## 7. Limites à mentionner clairement

Le projet reste académique. Il ne doit pas être présenté comme un système bancaire prêt pour la production.

Les principales limites sont :

- les datasets ne sont pas inclus dans le dépôt ;
- les fichiers du modèle entraîné ne sont pas inclus ;
- certaines parties du dashboard sont simulées si le modèle n'est pas disponible ;
- les notebooks doivent être rejoués pour reproduire entièrement les résultats ;
- les explications sont moins utiles sur Kaggle Credit Card, car les variables `V1` à `V28` sont anonymisées par ACP ;
- les résultats finaux doivent être harmonisés entre les différents fichiers CSV ;
- il manque encore des tests d'intégration complets pour l'API, le dashboard et les explications SHAP/LIME.

## 8. Ordre de lecture conseillé pour le tuteur

Pour comprendre rapidement le projet, l'ordre de lecture conseillé est :

1. `README.md` pour la vue générale.
2. `ANNEXE_TECHNIQUE_GUIDE_CODE.md` pour comprendre le fonctionnement pratique.
3. `config.py` pour voir les chemins et paramètres.
4. `src/data/preprocessor.py` pour comprendre la préparation des données.
5. `src/data/balancer.py` pour comprendre SMOTE et le déséquilibre.
6. `src/models/stacking_model.py` pour voir la contribution principale.
7. `src/evaluation/metrics.py` pour comprendre les métriques.
8. `src/explainability/shap_explainer.py` et `src/explainability/lime_explainer.py` pour l'explicabilité.
9. `src/api/routes.py` pour voir comment le modèle est exposé.
10. `dashboard/pages/` pour voir l'interface utilisateur.
11. `tests/` pour voir ce qui est automatiquement vérifié.

Cet ordre permet de lire le code comme une progression logique, depuis les données jusqu'à l'application.

## 9. Problèmes fréquents et solutions

### Problème : `No module named pytest`

Solution :

```bash
python -m pip install pytest
```

### Problème : `No module named catboost`

Solution :

```bash
python -m pip install catboost
```

CatBoost est importé dans `src/models/ensemble_models.py`. Même si l'on ne teste pas directement CatBoost, son absence peut bloquer l'import du module.

### Problème : `Model not loaded`

Cause probable : le fichier `models/stacking_ensemble.pkl` n'existe pas.

Solution : exécuter le notebook d'entraînement du stacking ou placer le modèle entraîné dans `models/`.

### Problème : dataset introuvable

Cause probable : `data/raw/creditcard.csv` ou `data/raw/paysim.csv` manque.

Solution : télécharger les datasets depuis Kaggle et les placer dans `data/raw/`.

### Problème : les résultats du dashboard changent ou semblent aléatoires

Cause probable : le dashboard est en mode démonstration car le modèle n'est pas chargé.

Solution : générer le modèle final et le scaler, puis relancer le dashboard.

## 10. Conclusion pratique

Le code représente la partie pratique du mémoire. Il ne se limite pas à quelques notebooks : il propose une structure complète avec modules de données, modèles, entraînement, évaluation, explicabilité, API, dashboard et tests.

La valeur principale du projet est pédagogique et expérimentale. Il montre comment transformer une problématique théorique de détection de fraude en prototype logiciel organisé. Pour un usage industriel, il faudrait encore renforcer la reproductibilité, supprimer les parties simulées, harmoniser les résultats et tester le système de bout en bout avec les vrais artefacts entraînés.

En l'état, le dépôt est donc correctement positionné comme un prototype académique démontrant la faisabilité de l'approche par stacking et explicabilité.
