"""
SETUP INSTRUCTIONS & QUICK START GUIDE
=====================================

Follow these steps to get the MTR-QNet Lite ML Service running:

STEP 1: INSTALL DEPENDENCIES
=============================

Option A - Windows:
  1. Open Command Prompt in ml-service folder
  2. Run: setup.bat
  3. This will automatically install all dependencies and train models

Option B - Linux/Mac:
  1. Open Terminal in ml-service folder
  2. Run: bash setup.sh
  3. This will automatically install all dependencies and train models

Option C - Manual:
  1. pip install -r requirements.txt


STEP 2: TRAIN MODELS (IMPORTANT!)
=================================

If you didn't use setup.bat/setup.sh, train models:

  python train_models.py

Expected Output:
  - Downloads 5 years of stock data
  - Trains 10 deep learning models (GRU+Attention+TCN)
  - Creates 'saved_models/' directory with trained models
  - Takes 10-15 minutes on a typical machine
  
Required disk space: ~100MB (for data) + ~50MB (for models)


STEP 3: START THE SERVICE
==========================

Windows:
  python -m uvicorn main:app --port 8001

Linux/Mac:
  python3 -m uvicorn main:app --port 8001

Or with auto-reload during development:
  python -m uvicorn main:app --port 8001 --reload


STEP 4: TEST THE SERVICE
=========================

Option A - Run Python test script:
  python test_models.py

Option B - Manual API testing:

  1. In PowerShell/Terminal, test the health endpoint:
     curl http://localhost:8001/health

  2. Test a prediction:
     $json = @{symbol="RELIANCE.NS"; name="Reliance Industries"} | ConvertTo-Json
     curl -X POST http://localhost:8001/predict `
       -H "Content-Type: application/json" `
       -d $json

  3. View available models:
     curl http://localhost:8001/models/status


EXPECTED PREDICTIONS
====================

When successfully trained, you should get predictions like:

{
  "symbol": "RELIANCE.NS",
  "trend": "UP",
  "confidence": 78.5,
  "atr": 42.34,
  "signal": "BUY",
  "priceRange": {
    "low": 2543.25,
    "high": 2589.89,
    "current": 2566.57
  },
  "volatility": {
    "move_pct": 0.91,
    "risk_level": "LOW"
  }
}

Key Outputs:
- signal: BUY/SELL/HOLD - Main trading signal
- trend: UP/DOWN/NEUTRAL - Market direction
- confidence: 0-100 - Model confidence percentage
- priceRange: Expected price movement range
- risk_level: LOW/MEDIUM/HIGH - Volatility estimate


INTEGRATION WITH BACKEND
=========================

Update backend/routes/predictions.js to call the ML service:

  // Example Node.js code
  const symbol = 'RELIANCE.NS';
  const response = await fetch('http://localhost:8001/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({symbol, name: 'Reliance Industries'})
  });
  const prediction = await response.json();
  
  res.json({
    signal: prediction.signal,
    trend: prediction.trend,
    confidence: prediction.confidence,
    priceRange: prediction.priceRange,
    timestamp: new Date()
  });


TROUBLESHOOTING
===============

Problem: "ModuleNotFoundError: No module named 'tensorflow'"
Solution: pip install tensorflow

Problem: "Models not loaded" error
Solution: You need to train models first
         Run: python train_models.py
         Wait for completion (10-15 minutes)

Problem: "Connection refused" when calling API
Solution: Make sure ML service is running
         Run: python -m uvicorn main:app --port 8001

Problem: Out of memory during training
Solution: Reduce batch_size in train_models.py
         Default is 32, try reducing to 16 or 8

Problem: Training is too slow
Solution: Enable GPU support (if CUDA available)
         Models will auto-detect and use GPU
         
Problem: Predictions don't match expected values
Solution: Retrain models with: python train_models.py
         Previous models cache may be affecting predictions


CUSTOMIZATION
=============

Change Training Ticker:
  Edit train_models.py, line 28:
  TICKER = "TCS.NS"  # or any other ticker
  Then retrain with: python train_models.py

Adjust Model Architecture:
  Edit train_models.py, search for "Hyperparameters"
  Modify: gru_units, tcn_filters, dropout_rate, etc.
  Then retrain

Change EMA Periods:
  Edit train_models.py, line 31:
  CANDIDATE_EMA_PAIRS = [(10,20), (12,26), ...]
  Then retrain


PROJECT FILES
=============

Main Files:
  - main.py              : FastAPI application (run this)
  - train_models.py      : Model training script (run once)
  - model_loader.py      : Model loading & prediction logic
  - test_models.py       : Validation tests
  - requirements.txt     : Python dependencies

Setup Scripts:
  - setup.bat            : Windows auto-setup
  - setup.sh             : Linux/Mac auto-setup

Configuration:
  - README.md            : Full documentation

Generated Files (after training):
  - saved_models/        : Directory with trained models
    - trend.pkl
    - vol.pkl
    - range.pkl
    - gap.pkl
    - vsp.pkl
    - mom.pkl
    - break.pkl
    - ez.pkl
    - ec.pkl
    - atr.pkl
    - metadata.pkl
    - scalings.pkl


API DOCUMENTATION
=================

Complete API reference available in README.md

Main endpoints:
  GET  /                 - Service info
  GET  /health           - Health check
  POST /predict          - Single prediction
  POST /predict/batch    - Multiple predictions
  GET  /models/status    - Model status


NEXT STEPS
==========

1. ✓ Install dependencies
2. ✓ Train models (train_models.py)
3. ✓ Start service (python -m uvicorn main:app --port 8001)
4. ✓ Test predictions (python test_models.py)
5. ✓ Integrate with backend (update routes/predictions.js)
6. ✓ Deploy to production

For more details, see README.md
"""

# This file is informational only
# See README.md for complete documentation
