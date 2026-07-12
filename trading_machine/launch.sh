#!/usr/bin/env bash
#
# launch.sh — One-command setup for the Autonomous Trading Machine.
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh
#
# This script:
#   1. Creates a Python 3.12 virtual environment
#   2. Installs all dependencies from requirements.txt
#   3. Creates a .env template for your FMP_API_KEY
#   4. Runs setup.py
#   5. Prints ready-to-use instructions
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}  AUTONOMOUS TRADING MACHINE — SETUP${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo ""

# ------------------------------------------------------------------
# Step 1: Find Python 3.12
# ------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Locating Python 3.12...${NC}"

PYTHON=""
for candidate in \
    "/opt/homebrew/bin/python3.12" \
    "/usr/local/bin/python3.12" \
    "/usr/bin/python3.12" \
    "python3.12" \
    "python3"; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        if [[ "$ver" == "3.12" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${RED}ERROR: Python 3.12 not found.${NC}"
    echo "  Install it via Homebrew:  brew install python@3.12"
    echo "  Or via pyenv:             pyenv install 3.12"
    exit 1
fi
echo -e "  Using: ${GREEN}$PYTHON${NC} ($($PYTHON --version))"

# ------------------------------------------------------------------
# Step 2: Create virtual environment
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    echo -e "  Virtual environment already exists at ${GREEN}.venv${NC}"
    echo -n "  Recreate? [y/N] "
    read -r RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo "  Removed old environment."
    else
        echo "  Keeping existing environment."
        SKIP_VENV=1
    fi
fi

if [[ -z "${SKIP_VENV:-}" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
    echo -e "  ${GREEN}Virtual environment created.${NC}"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo -e "  Activated: ${GREEN}$VENV_DIR${NC}"

# ------------------------------------------------------------------
# Step 3: Install dependencies
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/5] Installing dependencies...${NC}"

# Upgrade pip first
pip install --upgrade pip -q 2>&1 | tail -1

# Check for HDF5 (needed by tables package)
HDF5_DIR=""
if command -v brew &>/dev/null; then
    HDF5_PREFIX=$(brew --prefix hdf5 2>/dev/null || echo "")
    if [[ -n "$HDF5_PREFIX" && -d "$HDF5_PREFIX" ]]; then
        export HDF5_DIR="$HDF5_PREFIX"
        echo "  HDF5 found at: $HDF5_DIR"
    else
        echo "  Installing HDF5 via Homebrew..."
        brew install hdf5 2>&1 | tail -1
        HDF5_PREFIX=$(brew --prefix hdf5 2>/dev/null || echo "")
        if [[ -n "$HDF5_PREFIX" ]]; then
            export HDF5_DIR="$HDF5_PREFIX"
        fi
    fi
fi

echo "  Installing packages from requirements.txt..."
pip install -r requirements.txt 2>&1 | tail -5

echo -e "  ${GREEN}Dependencies installed.${NC}"

# ------------------------------------------------------------------
# Step 4: Create .env template
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/5] Setting up .env configuration...${NC}"

ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    echo -e "  ${GREEN}.env already exists.${NC}"
    # Check if FMP_API_KEY is set
    if grep -q "FMP_API_KEY=your_key_here" "$ENV_FILE" 2>/dev/null; then
        echo -e "  ${RED}WARNING: FMP_API_KEY still has placeholder value.${NC}"
        echo "  Edit .env and replace 'your_key_here' with your actual key."
    else
        KEY_VAL=$(grep "FMP_API_KEY=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- || echo "")
        if [[ -n "$KEY_VAL" ]]; then
            echo "  FMP_API_KEY is configured."
        fi
    fi
else
    cat > "$ENV_FILE" << 'ENVEOF'
# Autonomous Trading Machine — Environment Configuration
#
# Get your free API key at: https://financialmodelingprep.com/developer/docs/
# Then replace 'your_key_here' below with your actual key.

FMP_API_KEY=your_key_here

# Optional: override defaults
# TICKERS=SPY,TSLA,NVDA,AMZN,NFLX,CSCO,QQQ,IWM,SPX
# START_DATE=2018-01-01
# END_DATE=2026-07-10
# DASHBOARD_PORT=8501
ENVEOF
    echo -e "  ${GREEN}.env template created.${NC}"
    echo -e "  ${RED}IMPORTANT: Edit .env and set your FMP_API_KEY.${NC}"
fi

# Also export for current session if set in env
if [[ -n "${FMP_API_KEY:-}" ]]; then
    echo "  Using FMP_API_KEY from environment."
fi

# ------------------------------------------------------------------
# Step 5: Run setup.py
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[5/5] Running setup.py...${NC}"

pip install -e . -q 2>&1 | tail -1
echo -e "  ${GREEN}Package installed in development mode.${NC}"

# Create required directories
mkdir -p data/storage models/saved logs

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}  SETUP COMPLETE${NC}"
echo -e "${GREEN}${BOLD}============================================================${NC}"
echo ""
echo -e "  ${BOLD}System ready. Run:${NC}"
echo ""
echo -e "    ${BLUE}source .venv/bin/activate${NC}"
echo -e "    ${BLUE}export FMP_API_KEY=your_key_here${NC}"
echo -e "    ${BLUE}python train_all.py${NC}          ${GREEN}# Full training pipeline${NC}"
echo ""
echo -e "  After training:"
echo -e "    ${BLUE}python run.py dashboard${NC}      ${GREEN}# Launch Streamlit dashboard${NC}"
echo -e "    ${BLUE}python run.py live${NC}            ${GREEN}# Start live trading${NC}"
echo -e "    ${BLUE}python run.py status${NC}          ${GREEN}# Check system status${NC}"
echo ""
echo -e "  ${YELLOW}Need an FMP API key?${NC}"
echo -e "  Get one free at: https://financialmodelingprep.com/developer/docs/"
echo ""
