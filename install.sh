#!/bin/bash
set -e
echo ""
echo "  Qythera - Production Superintelligence"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  Error: Python 3 not found"
    exit 1
fi
echo "  Python: $(python3 --version)"

# Clone if not exists
if [ ! -f "core/__init__.py" ]; then
    echo "  Downloading Qythera..."
    git clone https://github.com/cpner/qythera.git qythera 2>/dev/null || {
        mkdir -p qythera && cd qythera
        curl -fsSL https://github.com/cpner/qythera/archive/main.tar.gz | tar xz --strip-components=1
        cd ..
    }
fi

# Enter project
if [ -d "qythera" ]; then
    cd qythera
elif [ -f "core/__init__.py" ]; then
    cd .
else
    echo "  Error: Could not find project"
    exit 1
fi

echo "  Location: $(pwd)"

# Install numpy
echo "  Installing numpy..."
pip3 install numpy -q 2>/dev/null || pip install numpy -q 2>/dev/null || python3 -m pip install numpy -q 2>/dev/null

echo ""
echo "  Starting Qythera server..."
echo "  Open in browser: http://localhost:8000"
echo ""
python3 -m core.inference.server
