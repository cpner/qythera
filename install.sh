#!/bin/bash
# Qythera - One-line installer
# Usage: curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/install.sh | bash

set -e

echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║       Q y t h e r a   A I         ║"
echo "  ║    Production Superintelligence    ║"
echo "  ╚═══════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $PYTHON_VERSION"

# Clone or use existing
if [ -d "qythera" ]; then
    echo "  Using existing qythera directory"
    cd qythera
else
    echo "  Cloning Qythera..."
    git clone https://github.com/cpner/qythera.git 2>/dev/null || {
        echo "  Downloading..."
        mkdir -p qythera && cd qythera
        curl -fsSL https://github.com/cpner/qythera/archive/main.tar.gz | tar xz --strip-components=1
    }
    cd qythera
fi

# Install numpy (only dependency)
echo "  Installing numpy..."
pip3 install numpy -q 2>/dev/null || pip install numpy -q 2>/dev/null || python3 -m pip install numpy -q 2>/dev/null

echo ""
echo "  ✅ Installed successfully!"
echo ""
echo "  Quick start:"
echo "    python3 -m inference.server    # Start AI server"
echo "    python3 cli/main.py chat       # Interactive chat"
echo "    cd web && npm install && npm run dev  # Web UI"
echo ""
echo "  Server will be at: http://localhost:8000"
echo "  Web UI will be at: http://localhost:3000"
echo ""

# Ask to start
read -p "  Start server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Starting Qythera server..."
    python3 -m inference.server
fi
