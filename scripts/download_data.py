"""Helper script to verify dataset availability."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW_DIR, KAGGLE_CC_PATH, PAYSIM_PATH


def check_datasets():
    """Check if required datasets are present."""
    print("=" * 60)
    print("Vérification des Jeux de Données")
    print("=" * 60)

    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # Kaggle Credit Card
    if os.path.exists(KAGGLE_CC_PATH):
        size_mb = os.path.getsize(KAGGLE_CC_PATH) / (1024 * 1024)
        print(f"[OK] Kaggle Credit Card: {KAGGLE_CC_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"[MANQUANT] Kaggle Credit Card")
        print(f"  Téléchargez depuis: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"  Placez le fichier dans: {KAGGLE_CC_PATH}")

    print()

    # PaySim
    if os.path.exists(PAYSIM_PATH):
        size_mb = os.path.getsize(PAYSIM_PATH) / (1024 * 1024)
        print(f"[OK] PaySim: {PAYSIM_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"[MANQUANT] PaySim")
        print(f"  Téléchargez depuis: https://www.kaggle.com/datasets/ealaxi/paysim1")
        print(f"  Placez le fichier dans: {PAYSIM_PATH}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    check_datasets()
