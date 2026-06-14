#!/bin/bash
set -e
echo "Installing Qythera..."
pip install -e .
cd web && npm install
echo "Done! Usage: qythera chat"
