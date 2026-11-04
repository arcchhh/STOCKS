@echo off
REM MTR-QNet Lite Setup Script for Windows
REM This script installs dependencies and trains models

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                                                       ║
echo ║  StockAI ML Service - MTR-QNet Lite Setup             ║
echo ║                                                       ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ✓ Dependencies installed successfully
echo.
echo [2/3] Creating saved_models directory...
if not exist "saved_models" mkdir saved_models
echo ✓ Directory created
echo.

echo [3/3] Training models (This will take 10-15 minutes)...
echo.
echo Note: This will download 5 years of stock data and train 10 deep learning models
echo Press Ctrl+C to cancel
echo.

python train_models.py
if errorlevel 1 (
    echo [ERROR] Training failed
    pause
    exit /b 1
)

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                                                       ║
echo ║  ✓ Setup Complete!                                    ║
echo ║                                                       ║
echo ║  To start the ML service, run:                       ║
echo ║  python -m uvicorn main:app --port 8001              ║
echo ║                                                       ║
echo ║  API will be available at: http://localhost:8001     ║
echo ║                                                       ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

pause
