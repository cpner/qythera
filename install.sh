#!/bin/bash
set -e
echo ""
echo "  Qythera - Production Superintelligence"
echo ""
if ! command -v python3 &> /dev/null; then echo "Error: Python 3 not found"; exit 1; fi
echo "  Python: $(python3 --version)"
if [ -d "qythera" ]; then cd qythera; else git clone https://github.com/cpner/qythera.git 2>/dev/null && cd qythera || { mkdir -p qythera && cd qythera && curl -fsSL https://github.com/cpner/qythera/archive/main.tar.gz | tar xz --strip-components=1; }; fi
pip3 install numpy -q 2>/dev/null || pip install numpy -q 2>/dev/null
echo "  Installed!"
echo "  Run: python -m core.inference.server"
