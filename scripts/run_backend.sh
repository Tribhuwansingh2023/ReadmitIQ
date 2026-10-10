#!/usr/bin/env bash
# ==============================================================================
# ReadmitIQ - Backend Launch Script (Linux / macOS)
# Starts the FastAPI clinical decision support service on http://localhost:8000
# ==============================================================================

set -euo pipefail

echo "===================================================================="
echo " ReadmitIQ Decision Support Backend Launcher (POSIX)"
echo "===================================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found on PATH. Please install Python 3.11 or 3.12."
    exit 1
fi

# Detect virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "[WARNING] No virtual environment found. Running with current Python."
fi

# Set default configuration
export READMITIQ_SERVE_MODE="${READMITIQ_SERVE_MODE:-full}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"

echo "[INFO] Starting Uvicorn server on ${API_HOST}:${API_PORT} (mode: ${READMITIQ_SERVE_MODE})..."
exec python3 -m uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}" --reload
