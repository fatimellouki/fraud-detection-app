#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PFE — Détection de Fraude : Lanceur de démonstration
# Usage : ./run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Vérifier que l'installation a été faite ─────────────────────────────────
if [ ! -d ".venv" ]; then
    echo -e "${RED}[ERREUR] L'environnement virtuel .venv n'existe pas.${NC}"
    echo "Exécutez d'abord :"
    echo -e "  ${GREEN}./install.sh${NC}"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ─── Menu principal ──────────────────────────────────────────────────────────
clear
echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}  PFE — Détection Intelligente de Fraude${NC}"
echo -e "${BLUE}  Démonstration interactive${NC}"
echo -e "${BLUE}=============================================================${NC}"
echo ""
echo "Choisissez ce que vous voulez lancer :"
echo ""
echo -e "  ${GREEN}1)${NC} Dashboard Streamlit  ${YELLOW}(recommandé pour une démo visuelle)${NC}"
echo -e "  ${GREEN}2)${NC} API Flask REST       ${YELLOW}(prédiction par requête HTTP)${NC}"
echo -e "  ${GREEN}3)${NC} Notebooks Jupyter    ${YELLOW}(analyse pas à pas, 11 notebooks)${NC}"
echo -e "  ${GREEN}4)${NC} Suite de tests       ${YELLOW}(pytest)${NC}"
echo -e "  ${GREEN}5)${NC} Vérifier les datasets"
echo -e "  ${GREEN}q)${NC} Quitter"
echo ""
read -rp "Votre choix [1-5/q] : " CHOICE
echo ""

case "$CHOICE" in
    1)
        echo -e "${BLUE}>>> Lancement du Dashboard Streamlit...${NC}"
        echo -e "${YELLOW}    Ouverture automatique sur http://localhost:8501${NC}"
        echo -e "${YELLOW}    Appuyez sur Ctrl+C pour arrêter.${NC}"
        echo ""
        streamlit run dashboard/app.py
        ;;
    2)
        echo -e "${BLUE}>>> Lancement de l'API Flask...${NC}"
        echo -e "${YELLOW}    Endpoint : POST http://localhost:5000/predict${NC}"
        echo -e "${YELLOW}    Appuyez sur Ctrl+C pour arrêter.${NC}"
        echo ""
        python -m src.api.app
        ;;
    3)
        echo -e "${BLUE}>>> Lancement de Jupyter Notebook...${NC}"
        if ! command -v jupyter &> /dev/null; then
            echo -e "${YELLOW}[INFO] Installation de jupyter...${NC}"
            pip install jupyter --quiet
        fi
        cd notebooks/
        jupyter notebook
        ;;
    4)
        echo -e "${BLUE}>>> Exécution de la suite de tests...${NC}"
        echo ""
        pytest tests/ -v
        ;;
    5)
        echo -e "${BLUE}>>> Vérification des datasets...${NC}"
        echo ""
        python scripts/download_data.py
        ;;
    q|Q)
        echo "Au revoir."
        exit 0
        ;;
    *)
        echo -e "${RED}[ERREUR] Choix invalide.${NC}"
        exit 1
        ;;
esac
