#!/bin/bash
set -e

echo ""
echo "  \033[35m╔══════════════════════════════════╗\033[0m"
echo "  \033[35m║   Qythera - AI Superintelligence  ║\033[0m"
echo "  \033[35m╚══════════════════════════════════╝\033[0m"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  \033[31mERROR: Python 3 not found\033[0m"
    echo "  Install Python: https://www.python.org/downloads/"
    exit 1
fi
echo "  \033[32m✓\033[0m Python: $(python3 --version)"

# Detect environment
if [ -f "/etc/serv00-release" ] || [ -d "/usr/home" ]; then
    echo "  \033[36mDetected: serv00/FreeBSD hosting\033[0m"
    PORT=8080
elif [ -d "/data/data/com.termux" ]; then
    echo "  \033[36mDetected: Termux (Android)\033[0m"
    PORT=8080
elif [ -d "/storage/emulated/0" ]; then
    echo "  \033[36mDetected: Android storage\033[0m"
    PORT=8080
else
    PORT=8000
fi

# Find project
if [ -f "core/__init__.py" ]; then
    PROJDIR="."
elif [ -d "qythera" ] && [ -f "qythera/core/__init__.py" ]; then
    PROJDIR="qythera"
else
    echo "  \033[36mDownloading Qythera...\033[0m"
    git clone https://github.com/cpner/qythera.git /tmp/qythera_download 2>/dev/null
    PROJDIR="/tmp/qythera_download"
fi

cd "$PROJDIR"
echo "  \033[32m✓\033[0m Location: $(pwd)"

# Install numpy
echo "  \033[36mInstalling numpy...\033[0m"
pip3 install numpy -q 2>/dev/null || pip install numpy -q 2>/dev/null || python3 -m pip install numpy -q --user 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  \033[31mERROR: Could not install numpy\033[0m"
    echo "  Try: pip3 install numpy --user"
    exit 1
fi

echo "  \033[32m✓\033[0m numpy installed"
echo ""
echo "  \033[35mStarting Qythera server...\033[0m"
echo "  \033[36mOpen in browser: http://localhost:$PORT\033[0m"
echo ""
python3 -m core.inference.server --port $PORT
