#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PFE — Détection de Fraude : Script d'installation automatique
# Usage : ./install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}  PFE — Détection de Fraude : Installation${NC}"
echo -e "${BLUE}=============================================================${NC}"
echo ""

# ─── 1. Vérifier Python 3.10+ ────────────────────────────────────────────────
echo -e "${BLUE}[1/4] Vérification de Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERREUR] Python 3 n'est pas installé.${NC}"
    echo "Installez Python 3.10+ depuis : https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo -e "${RED}[ERREUR] Python 3.10+ requis (détecté : $PYTHON_VERSION).${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Python $PYTHON_VERSION détecté${NC}"
echo ""

# ─── 2. Créer l'environnement virtuel ────────────────────────────────────────
echo -e "${BLUE}[2/4] Création de l'environnement virtuel (.venv)...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}[INFO] .venv existe déjà — réutilisation${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}[OK] .venv créé${NC}"
fi
echo ""

# shellcheck disable=SC1091
source .venv/bin/activate

# ─── 3. Installer les dépendances ────────────────────────────────────────────
echo -e "${BLUE}[3/4] Installation des dépendances (peut prendre 5–10 minutes)...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}[OK] Toutes les dépendances sont installées${NC}"
echo ""

# ─── 4. Vérifier / télécharger les datasets ──────────────────────────────────
echo -e "${BLUE}[4/4] Vérification des jeux de données...${NC}"
mkdir -p data/raw data/processed models reports/figures

CC_PATH="data/raw/creditcard.csv"
PAYSIM_PATH="data/raw/paysim.csv"

download_with_kaggle() {
    local DATASET="$1"
    local OUTDIR="$2"
    if ! command -v kaggle &> /dev/null; then
        return 1
    fi
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        return 1
    fi
    kaggle datasets download -d "$DATASET" -p "$OUTDIR" --unzip --quiet
    return $?
}

NEED_MANUAL=0

if [ -f "$CC_PATH" ]; then
    echo -e "${GREEN}[OK] creditcard.csv déjà présent${NC}"
else
    echo -e "${YELLOW}[INFO] creditcard.csv manquant — tentative via Kaggle API...${NC}"
    if download_with_kaggle "mlg-ulb/creditcardfraud" "data/raw"; then
        echo -e "${GREEN}[OK] creditcard.csv téléchargé${NC}"
    else
        echo -e "${YELLOW}[INFO] Kaggle API non configurée — téléchargement manuel requis${NC}"
        NEED_MANUAL=1
    fi
fi

if [ -f "$PAYSIM_PATH" ]; then
    echo -e "${GREEN}[OK] paysim.csv déjà présent${NC}"
else
    echo -e "${YELLOW}[INFO] paysim.csv manquant — tentative via Kaggle API...${NC}"
    if download_with_kaggle "ealaxi/paysim1" "data/raw"; then
        echo -e "${GREEN}[OK] paysim.csv téléchargé${NC}"
    else
        echo -e "${YELLOW}[INFO] Kaggle API non configurée — téléchargement manuel requis${NC}"
        NEED_MANUAL=1
    fi
fi

echo ""
echo -e "${BLUE}=============================================================${NC}"
if [ $NEED_MANUAL -eq 1 ]; then
    echo -e "${YELLOW}  Installation terminée — datasets à télécharger manuellement${NC}"
    echo -e "${BLUE}=============================================================${NC}"
    echo ""
    echo "Téléchargez les datasets et placez-les dans data/raw/ :"
    echo ""
    echo "  1. creditcard.csv"
    echo "     https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo ""
    echo "  2. paysim.csv"
    echo "     https://www.kaggle.com/datasets/ealaxi/paysim1"
    echo ""
    echo "Astuce : pour automatiser, configurez la Kaggle API :"
    echo "  pip install kaggle"
    echo "  → puis placez votre kaggle.json dans ~/.kaggle/"
    echo "  → relancez ./install.sh"
else
    echo -e "${GREEN}  Installation terminée avec succès !${NC}"
    echo -e "${BLUE}=============================================================${NC}"
    echo ""
    echo "Lancez la démonstration avec :"
    echo -e "  ${GREEN}./run.sh${NC}"
fi
echo ""
