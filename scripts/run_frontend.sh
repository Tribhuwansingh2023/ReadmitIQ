#!/usr/bin/env bash
# ==============================================================================
# ReadmitIQ - Frontend Launch Script (Linux / macOS)
# Starts the Vite development server on http://localhost:5173
# ==============================================================================

set -euo pipefail

echo "===================================================================="
echo " ReadmitIQ React/Vite Frontend Launcher (POSIX)"
echo "===================================================================="

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "[ERROR] npm is not found on PATH. Please install Node.js 18+."
    exit 1
fi

cd web

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[INFO] Installing frontend dependencies with npm install..."
    npm install
fi

echo "[INFO] Starting Vite development server on http://localhost:5173..."
exec npm run dev
