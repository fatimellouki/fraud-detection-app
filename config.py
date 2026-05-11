"""Global configuration constants for the fraud detection project."""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# ─── Dataset Configuration ──────────────────────────────────────────────────
KAGGLE_CC_FILENAME = "creditcard.csv"
PAYSIM_FILENAME = "paysim.csv"

KAGGLE_CC_PATH = os.path.join(DATA_RAW_DIR, KAGGLE_CC_FILENAME)
PAYSIM_PATH = os.path.join(DATA_RAW_DIR, PAYSIM_FILENAME)

# ─── Feature Configuration ──────────────────────────────────────────────────
KAGGLE_PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
KAGGLE_OTHER_FEATURES = ["Amount", "Time"]
KAGGLE_ALL_FEATURES = KAGGLE_PCA_FEATURES + KAGGLE_OTHER_FEATURES
TARGET_COL = "Class"
FRAUD_LABEL = 1
LEGIT_LABEL = 0

PAYSIM_FEATURES = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
]
PAYSIM_TARGET = "isFraud"

# ─── Splitting ───────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

# ─── Cross-Validation ───────────────────────────────────────────────────────
CV_FOLDS = 5

# ─── Model Hyperparameters (defaults) ───────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

DECISION_TREE_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 5,
    "criterion": "gini",
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "is_unbalance": True,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "auto_class_weights": "Balanced",
    "verbose": 0,
    "random_state": RANDOM_STATE,
}

MLP_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 256,
}

AUTOENCODER_PARAMS = {
    "encoding_dim": 8,
    "hidden_dims": [16],
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 0.001,
}

STACKING_META_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

# ─── Visualization ──────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_GRID = (16, 12)
FIGURE_FORMAT = "png"
FONT_SIZE = 12

# ─── Cost-Benefit Analysis ──────────────────────────────────────────────────
AVG_FRAUD_AMOUNT = 122.0  # euros (from Kaggle CC data)
FP_COST = 5.0             # euros (customer service per false alert)
INVESTIGATION_COST = 2.0  # euros (cost to review flagged transaction)

# ─── API Configuration ──────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
DEFAULT_THRESHOLD = 0.5
