@echo off
REM ==============================================================================
REM ReadmitIQ - Frontend Launch Script (Windows)
REM Starts the Vite development server on http://localhost:5173
REM ==============================================================================

setlocal enabledelayedexpansion

echo ====================================================================
echo  ReadmitIQ React/Vite Frontend Launcher
echo ====================================================================

REM Check if npm is installed
npm --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm is not found on PATH. Please install Node.js 18+.
    exit /b 1
)

cd web

REM Check if node_modules exists
if not exist "node_modules\" (
    echo [INFO] Installing frontend dependencies with npm install...
    call npm install
)

echo [INFO] Starting Vite development server on http://localhost:5173...
npm run dev
