"""Entraînement et évaluation de la chaîne complète de détection de fraude.

Ce script reproduit la méthodologie du mémoire sur le jeu Kaggle Credit Card :

    1. Chargement des données       (src/data/loader.py)
    2. Découpage stratifié          (src/data/splitter.py)
    3. Normalisation StandardScaler (src/data/preprocessor.py)
    4. Rééquilibrage SMOTE          (src/data/balancer.py)
    5. Entraînement de 8 modèles    (src/models/*)
    6. Évaluation sur le test       (src/evaluation/metrics.py)
    7. Sauvegarde du modèle final + des résultats réels

Méthodologie d'évaluation : les jeux train / validation / test sont disjoints.
Chaque modèle est entraîné sur le train (rééquilibré par SMOTE), son SEUIL de
décision est choisi sur la VALIDATION (maximisation du F1), puis les métriques
sont mesurées sur le TEST jamais vu. L'AUC-ROC et l'AUPRC ne dépendent pas du
seuil ; le F1, la précision et le rappel sont donnés au seuil optimal.

Sorties :
    models/stacking_ensemble.pkl   -> modèle utilisé par le dashboard et l'API
    models/scaler.pkl              -> normaliseur appliqué avant chaque prédiction
    models/final_results.csv       -> métriques détaillées (réelles)
    models/results_comparison.csv  -> tableau comparatif affiché par le dashboard
    data/samples/demo_transactions.csv -> exemples réels (fraude + légitime)

Usage :
    python scripts/train_model.py            # jeu complet
    python scripts/train_model.py --sample 50000   # sous-échantillon (plus rapide)

L'ordre des colonnes (V1..V28, Amount, Time) est celui attendu par le dashboard
et l'API : ne pas le modifier.
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from config import (
    KAGGLE_ALL_FEATURES, TARGET_COL, MODELS_DIR, RANDOM_STATE,
    MLP_PARAMS, AUTOENCODER_PARAMS,
)
from src.data.loader import load_creditcard
from src.data.splitter import stratified_split
from src.data.balancer import ImbalanceHandler
from src.models.base_models import create_logistic_regression, create_decision_tree
from src.models.ensemble_models import (
    create_random_forest, create_xgboost, create_lightgbm,
)
from src.models.stacking_model import FraudStackingEnsemble
from src.evaluation.metrics import compute_all_metrics


def log(msg):
    """Affiche un message horodaté."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_f1_threshold(y_true, scores):
    """Renvoie le seuil qui maximise le F1 sur l'ensemble de validation."""
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    # precision_recall_curve renvoie un point de plus que thresholds
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def measure_inference_ms(predict_fn, X_sample, n_runs=50):
    """Temps d'inférence pour UNE transaction, en millisecondes (latence réelle).

    On mesure la prédiction d'une seule transaction (le cas d'usage temps réel),
    pas une moyenne amortie sur un lot vectorisé. On renvoie la médiane sur
    plusieurs exécutions pour lisser le bruit de mesure.
    """
    one = X_sample[:1]
    predict_fn(one)  # warm-up (chargement paresseux, caches)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(one)
        times.append(time.perf_counter() - start)
    return round(float(np.median(times)) * 1000, 3)


def evaluate(name, y_true, proba_test, threshold, inference_ms):
    """Calcule toutes les métriques au seuil choisi et renvoie une ligne."""
    pred = (proba_test >= threshold).astype(int)
    m = compute_all_metrics(y_true, pred, proba_test)
    log(f"  {name:<22} AUC={m['auc_roc']:.4f}  AUPRC={m['auprc']:.4f}  "
        f"F1={m['f1_score']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
        f"(seuil={threshold:.3f})")
    return {
        "Modèle": name,
        "AUC-ROC": round(m["auc_roc"], 4),
        "AUPRC": round(m["auprc"], 4),
        "F1-Score": round(m["f1_score"], 4),
        "Précision": round(m["precision"], 4),
        "Rappel": round(m["recall"], 4),
        "Temps (ms)": inference_ms,
        "Seuil": round(threshold, 4),
        "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Nombre de lignes (0 = jeu complet).")
    parser.add_argument("--smote-strategy", type=float, default=0.1,
                        help="Ratio minorité/majorité visé par SMOTE.")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Chargement ────────────────────────────────────────────────────────
    log("1/7  Chargement du jeu Kaggle Credit Card...")
    df = load_creditcard()
    if args.sample and args.sample < len(df):
        fraud = df[df[TARGET_COL] == 1]
        legit = df[df[TARGET_COL] == 0].sample(
            n=max(args.sample - len(fraud), 1), random_state=RANDOM_STATE)
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_STATE)
        log(f"     Sous-échantillon : {len(df):,} lignes ({len(fraud):,} fraudes)")

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    X = df[KAGGLE_ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    # ── 2. Découpage stratifié train / val / test (disjoints) ────────────────
    log("2/7  Découpage stratifié train/val/test...")
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_split(X, y)

    # ── 3. Normalisation (fit sur le train uniquement) ───────────────────────
    log("3/7  Normalisation StandardScaler (fit sur train)...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)
    y_tr_a, y_val_a, y_te_a = y_tr.to_numpy(), y_val.to_numpy(), y_te.to_numpy()

    # ── 4. Rééquilibrage SMOTE (train normalisé) ─────────────────────────────
    log(f"4/7  SMOTE (sampling_strategy={args.smote_strategy})...")
    handler = ImbalanceHandler(random_state=RANDOM_STATE)
    X_res, y_res = handler.apply_smote(
        X_tr_s, y_tr_a, sampling_strategy=args.smote_strategy)
    log(f"     Entraînement : {len(y_res):,} lignes "
        f"({int((y_res == 1).sum()):,} fraudes après SMOTE)")

    results = []
    sample_eval = X_te_s[: min(2000, len(X_te_s))]

    # ── 5+6. Entraînement et évaluation ──────────────────────────────────────
    log("5/7  Entraînement des modèles...")

    classic = [
        ("Rég. Logistique", create_logistic_regression()),
        ("Arbre de Décision", create_decision_tree()),
        ("Random Forest", create_random_forest()),
        ("XGBoost", create_xgboost()),
        ("LightGBM", create_lightgbm()),
    ]
    for name, model in classic:
        try:
            model.fit(X_res, y_res)
            t = best_f1_threshold(y_val_a, model.predict_proba(X_val_s)[:, 1])
            proba_te = model.predict_proba(X_te_s)[:, 1]
            ms = measure_inference_ms(lambda x: model.predict_proba(x), sample_eval)
            results.append(evaluate(name, y_te_a, proba_te, t, ms))
        except Exception as e:
            log(f"  ÉCHEC {name}: {e}")

    # MLP (perceptron multicouche scikit-learn — pas de TensorFlow requis)
    try:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(MLP_PARAMS["hidden_layers"]),
            alpha=1e-4, learning_rate_init=MLP_PARAMS["learning_rate"],
            max_iter=40, early_stopping=True, random_state=RANDOM_STATE)
        mlp.fit(X_res, y_res)
        t = best_f1_threshold(y_val_a, mlp.predict_proba(X_val_s)[:, 1])
        proba_te = mlp.predict_proba(X_te_s)[:, 1]
        ms = measure_inference_ms(lambda x: mlp.predict_proba(x), sample_eval)
        results.append(evaluate("MLP", y_te_a, proba_te, t, ms))
    except Exception as e:
        log(f"  ÉCHEC MLP: {e}")

    # Auto-encodeur linéaire (ACP) : score d'anomalie = erreur de reconstruction,
    # appris sur les transactions légitimes uniquement (non supervisé).
    try:
        from sklearn.decomposition import PCA
        enc_dim = AUTOENCODER_PARAMS["encoding_dim"]
        pca = PCA(n_components=enc_dim, random_state=RANDOM_STATE)
        pca.fit(X_tr_s[y_tr_a == 0])

        def recon_err(Xs):
            r = pca.inverse_transform(pca.transform(Xs))
            return np.mean((Xs - r) ** 2, axis=1)

        err_val, err_te = recon_err(X_val_s), recon_err(X_te_s)
        lo, hi = err_val.min(), err_val.max()
        proba_te = np.clip((err_te - lo) / (hi - lo + 1e-12), 0, 1)
        # seuil sur l'erreur de validation, ramené à l'échelle proba
        t_err = best_f1_threshold(y_val_a, err_val)
        t = float(np.clip((t_err - lo) / (hi - lo + 1e-12), 0, 1))
        ms = measure_inference_ms(
            lambda x: pca.inverse_transform(pca.transform(x)), sample_eval)
        results.append(evaluate("Auto-encodeur", y_te_a, proba_te, t, ms))
    except Exception as e:
        log(f"  ÉCHEC Auto-encodeur: {e}")

    # Stacking (contribution principale) — RF + XGBoost + LightGBM -> méta-XGBoost
    log("     Entraînement du STACKING (plusieurs minutes)...")
    try:
        stack = FraudStackingEnsemble()
        stack.fit(X_res, y_res)
        t = best_f1_threshold(y_val_a, stack.predict_proba(X_val_s)[:, 1])
        proba_te = stack.predict_proba(X_te_s)[:, 1]
        ms = measure_inference_ms(lambda x: stack.predict_proba(x), sample_eval)
        results.append(evaluate("Stacking (proposé)", y_te_a, proba_te, t, ms))

        # ── 7. Sauvegarde modèle de production + scaler ─────────────────────
        log("7/7  Sauvegarde du modèle et du scaler...")
        stack.save(os.path.join(MODELS_DIR, "stacking_ensemble.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    except Exception as e:
        log(f"  ÉCHEC Stacking: {e}")

    # ── Résultats réels ──────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(MODELS_DIR, "final_results.csv"), index=False)
    display_cols = ["Modèle", "AUC-ROC", "AUPRC", "F1-Score",
                    "Précision", "Rappel", "Temps (ms)"]
    df_res[display_cols].to_csv(
        os.path.join(MODELS_DIR, "results_comparison.csv"), index=False)

    # ── Exemples réels pour la démonstration ─────────────────────────────────
    try:
        samples_dir = os.path.join(os.path.dirname(MODELS_DIR), "data", "samples")
        os.makedirs(samples_dir, exist_ok=True)
        te_idx = list(X_te.index)
        fraud_idx = [i for i in te_idx if y.loc[i] == 1][:5]
        legit_idx = [i for i in te_idx if y.loc[i] == 0][:10]
        sample_df = df.loc[fraud_idx + legit_idx,
                           KAGGLE_ALL_FEATURES + [TARGET_COL]].copy()
        sample_df.to_csv(os.path.join(samples_dir, "demo_transactions.csv"),
                         index=False)
        log(f"Exemples de démo : {len(sample_df)} transactions réelles "
            f"({len(fraud_idx)} fraudes, {len(legit_idx)} légitimes)")
    except Exception as e:
        log(f"  (Exemples de démo non générés : {e})")

    log(f"TERMINÉ en {time.time() - t0:.0f}s")
    print("\n" + df_res[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
