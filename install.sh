#!/bin/bash
# Qythera Universal Installer - works on Linux, macOS, FreeBSD, Android, Raspberry Pi

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${PURPLE}╔══════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║     Qythera - AI Superintelligence    ║${NC}"
echo -e "${PURPLE}║       Universal Installer v1.0        ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════╝${NC}"
echo ""

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"
PYTHON=""
PIP=""

echo -e "${CYAN}Detecting system...${NC}"

# Find Python
for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python 3 not found!${NC}"
    echo -e "${YELLOW}Install Python:${NC}"
    echo "  Linux:  sudo apt install python3  (Ubuntu/Debian)"
    echo "          sudo pacman -S python     (Arch)"
    echo "          sudo dnf install python3  (Fedora)"
    echo "  macOS:  brew install python3      (Homebrew)"
    echo "  FreeBSD: pkg install python3      (pkg)"
    echo "  Android: pkg install python        (Termux)"
    echo "  Windows: winget install Python.Python.3"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# Find pip
for cmd in pip3.14 pip3.13 pip3.12 pip3.11 pip3.10 pip3 pip; do
    if command -v "$cmd" &>/dev/null; then
        PIP="$cmd"
        break
    fi
done

if [ -z "$PIP" ]; then
    PIP="$PYTHON -m pip"
fi

# Detect environment
ENV="unknown"
if [ -f "/etc/serv00-release" ] || [ -d "/usr/home" ]; then
    ENV="serv00"
    echo -e "${CYAN}Environment: serv00/FreeBSD hosting${NC}"
elif [ -d "/data/data/com.termux" ]; then
    ENV="termux"
    echo -e "${CYAN}Environment: Termux (Android)${NC}"
elif [ -f "/etc/os-release" ]; then
    DISTRO=$(grep "^NAME=" /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"')
    ENV="$DISTRO"
    echo -e "${CYAN}Environment: $DISTRO${NC}"
elif [ "$OS" = "Darwin" ]; then
    ENV="macos"
    echo -e "${CYAN}Environment: macOS${NC}"
fi

# Find or clone project
echo ""
echo -e "${CYAN}Setting up Qythera...${NC}"

if [ -f "core/__init__.py" ]; then
    PROJDIR="$(pwd)"
    echo -e "${GREEN}✓ Found project in current directory${NC}"
elif [ -d "qythera" ] && [ -f "qythera/core/__init__.py" ]; then
    PROJDIR="$(pwd)/qythera"
    cd qythera
    echo -e "${GREEN}✓ Found project in ./qythera${NC}"
else
    echo -e "${YELLOW}Downloading Qythera...${NC}"
    git clone https://github.com/cpner/qythera.git /tmp/qythera_install 2>/dev/null
    PROJDIR="/tmp/qythera_install"
    cd /tmp/qythera_install
    echo -e "${GREEN}✓ Downloaded to $PROJDIR${NC}"
fi

# Install numpy
echo ""
echo -e "${CYAN}Installing numpy...${NC}"
$PIP install numpy --quiet 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Trying with --user flag...${NC}"
    $PIP install numpy --quiet --user 2>/dev/null
fi
echo -e "${GREEN}✓ numpy installed${NC}"

# Find free port
PORT=8080
if [ "$ENV" = "serv00" ]; then
    PORT=8080
elif [ "$ENV" = "termux" ]; then
    PORT=8080
else
    PORT=8000
fi

echo ""
echo -e "${PURPLE}╔══════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║         Installation Complete!        ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Quick start:${NC}"
echo -e "  ${CYAN}python3 -m core.inference.server --port $PORT${NC}"
echo ""
echo -e "${GREEN}Or use the standalone HTML (no server needed):${NC}"
echo -e "  ${CYAN}Open web/standalone.html in any browser${NC}"
echo ""
echo -e "${GREEN}Web UI:${NC}"
echo -e "  ${CYAN}cd web && npm install && npm run dev${NC}"
echo ""
echo -e "${GREEN}CLI:${NC}"
echo -e "  ${CYAN}python3 cli/main.py chat${NC}"
echo ""

# Ask to start
read -p "$(echo -e ${YELLOW}Start server now? [y/N]: ${NC})" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}Starting server on port $PORT...${NC}"
    python3 -m core.inference.server --port $PORT
fi
