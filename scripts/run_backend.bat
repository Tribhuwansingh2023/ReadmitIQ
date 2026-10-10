@echo off
REM ==============================================================================
REM ReadmitIQ - Backend Launch Script (Windows)
REM Starts the FastAPI clinical decision support service on http://localhost:8000
REM ==============================================================================

setlocal enabledelayedexpansion

echo ====================================================================
echo  ReadmitIQ Decision Support Backend Launcher
echo ====================================================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found on PATH. Please install Python 3.11 or 3.12.
    exit /b 1
)

REM Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] No virtual environment found. Running with global python.
)

REM Set default environment variables if not present
if "%READMITIQ_SERVE_MODE%"=="" set READMITIQ_SERVE_MODE=full
if "%API_HOST%"=="" set API_HOST=0.0.0.0
if "%API_PORT%"=="" set API_PORT=8000

echo [INFO] Starting Uvicorn server on %API_HOST%:%API_PORT% (mode: %READMITIQ_SERVE_MODE%)...
python -m uvicorn api.main:app --host %API_HOST% --port %API_PORT% --reload
