#!/bin/bash
# MTR-QNet Lite Setup Script for Linux/Mac
# This script installs dependencies and trains models

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                                                       ║"
echo "║  StockAI ML Service - MTR-QNet Lite Setup             ║"
echo "║                                                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ using your package manager"
    exit 1
fi

echo "[1/3] Installing Python dependencies..."
echo ""
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo ""
echo "✓ Dependencies installed successfully"
echo ""
echo "[2/3] Creating saved_models directory..."
mkdir -p saved_models
echo "✓ Directory created"
echo ""

echo "[3/3] Training models (This will take 10-15 minutes)..."
echo ""
echo "Note: This will download 5 years of stock data and train 10 deep learning models"
echo "Press Ctrl+C to cancel"
echo ""

python3 train_models.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed"
    exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                                                       ║"
echo "║  ✓ Setup Complete!                                    ║"
echo "║                                                       ║"
echo "║  To start the ML service, run:                       ║"
echo "║  python3 -m uvicorn main:app --port 8001             ║"
echo "║                                                       ║"
echo "║  API will be available at: http://localhost:8001     ║"
echo "║                                                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
