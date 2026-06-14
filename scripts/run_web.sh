#!/bin/bash
cd "$(dirname "$0")/../web"
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi
npm run dev
