#!/bin/bash
set -e

echo "╔══════════════════════════════════════╗"
echo "║     Qythera Installation Script      ║"
echo "╚══════════════════════════════════════╝"
echo ""

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected: $OS $ARCH"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PYTHON_VERSION"

# Check Docker
if command -v docker &> /dev/null; then
    echo "Docker: $(docker --version | head -1)"
else
    echo "Docker: not found (optional)"
fi

# Check NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "NVIDIA GPU: not found (CPU mode available)"
fi

echo ""
echo "Installing Qythera..."

# Create virtual environment
python3 -m venv .venv 2>/dev/null || true
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || pip install torch
pip install transformers accelerate peft datasets
pip install sentencepiece tiktoken
pip install faiss-cpu 2>/dev/null || true
pip install sentence-transformers 2>/dev/null || true
pip install click rich requests pyyaml
pip install fastapi uvicorn 2>/dev/null || true

# Install CLI
pip install -e . 2>/dev/null || true

# Create symlink for CLI
if [ ! -f "/usr/local/bin/qythera" ] && [ -f "cli/qythera" ]; then
    ln -sf "$(pwd)/cli/qythera" /usr/local/bin/qythera 2>/dev/null || true
fi

echo ""
echo "Installation complete!"
echo ""
echo "Quick start:"
echo "  qythera info       - Show system info"
echo "  qythera chat       - Start chat"
echo "  qythera serve      - Start inference server"
echo "  qythera web        - Launch web UI"
echo ""
echo "Or run directly:"
echo "  python3 cli/main.py chat"
echo "  python3 cli/main.py serve"
echo "  python3 -m inference.server"
