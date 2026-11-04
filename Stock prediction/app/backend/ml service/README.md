# MTR-QNet Lite ML Service

AI-powered stock market prediction using GRU+Attention+TCN Hybrid Deep Learning Models

## Features

- **Auto-Adaptive EMA Selection**: Automatically selects the best EMA periods for the given stock
- **Hybrid Deep Learning**: GRU with Attention + Temporal Convolutional Network (TCN)
- **Real Cross Events**: Detects actual EMA crossovers between trading days
- **Trend Zone Analysis**: Identifies bullish/bearish EMA zones
- **Volatility Prediction**: ATR-based price range forecasting
- **Ensemble Signals**: Combines 10 different prediction models for robust signals

## Installation & Setup

### 1. Install Dependencies

```bash
cd ml-service
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)

Run this ONCE to train and save all models:

```bash
python train_models.py
```

**Expected Output:**
- Downloads 5 years of stock data for RELIANCE.NS
- Trains 10 deep learning models (GRU+TCN hybrid)
- Saves models to `saved_models/` directory
- Process takes 10-15 minutes

**Output Files:**
- `saved_models/trend.pkl` - Trend regime classifier
- `saved_models/vol.pkl` - Volatility regime classifier
- `saved_models/range.pkl` - Price range predictor
- `saved_models/gap.pkl` - Gap direction predictor
- `saved_models/vsp.pkl` - Volume spike detector
- `saved_models/mom.pkl` - Momentum continuation classifier
- `saved_models/break.pkl` - Breakout detector
- `saved_models/ez.pkl` - EMA Zone analyzer
- `saved_models/ec.pkl` - EMA Crossover detector
- `saved_models/atr.pkl` - ATR (volatility) regressor
- `saved_models/metadata.pkl` - Model metadata
- `saved_models/scalings.pkl` - Feature scalings

### 3. Start the Service

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Or without reload:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### GET `/` - Service Info
```bash
curl http://localhost:8001/
```

### GET `/health` - Health Check
```bash
curl http://localhost:8001/health
```

### POST `/predict` - Get Stock Prediction
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "RELIANCE.NS", "name": "Reliance Industries"}'
```

**Response:**
```json
{
  "symbol": "RELIANCE.NS",
  "trend": "UP",
  "confidence": 85.5,
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
  },
  "indicators": {
    "trend_regime": 1,
    "ema_zone": 1,
    "ema_cross": 0,
    "momentum": 1,
    "breakout": 0,
    "gap": 1
  }
}
```

### POST `/predict/batch` - Batch Predictions
```bash
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '["RELIANCE.NS", "TCS.NS", "INFY.NS"]'
```

### GET `/models/status` - Model Status
```bash
curl http://localhost:8001/models/status
```

## Model Architecture

### Input Features (13)
- EMA(fast), EMA(slow), EMA_diff
- MACD, MACD_Signal, MACD_Hist
- RSI(14)
- Bollinger Bands Width
- ATR(14)
- Close Lag(1)
- Daily Return
- Volume Median(20)
- 20-day High

### Processing Pipeline
1. **Sequence Building**: 30-day sequences (SEQ_LEN)
2. **Normalization**: Z-score standardization
3. **Hybrid Architecture**:
   - **Branch A**: GRU(64) + Attention with Softmax
   - **Branch B**: TCN with 3 dilated convolutions (1, 2, 4)
   - **Fusion**: Concatenate + Dense layers + Softmax/Linear

### Output Targets
- Trend Regime: {-1: DOWN, 0: NEUTRAL, 1: UP}
- Volatility Regime: {0: LOW, 1: HIGH}
- Range Bucket: {0: SMALL, 1: MEDIUM, 2: LARGE}
- Gap Direction: {0: DOWN, 1: NEUTRAL, 2: UP}
- Volume Spike: {0: No, 1: Yes}
- Momentum Continue: {0: Reverse, 1: Continue}
- Breakout: {0: No, 1: Yes}
- EMA Zone: {-1: BEARISH, 0: NEUTRAL, 1: BULLISH}
- EMA Crossover: {-1: Bearish Cross, 0: No Cross, 1: Bullish Cross}
- ATR (Regression): Actual volatility value

## Signal Generation

### Buy Score Calculation
```
BUY_SCORE = 
  + 1.3 × TREND_UP
  + 0.9 × MOMENTUM_UP
  + 0.6 × VOL_LOW
  + 0.7 × RANGE_BIG
  + 1.0 × BREAKOUT
  + 0.7 × GAP_UP
  - 0.4 × GAP_DOWN
  - 0.9 × VOLUME_SPIKE
  - ATR_NORMALIZED
  + 0.35 × EMA_ZONE_BULL
  - 0.35 × EMA_ZONE_BEAR
  + 0.8 × EMA_CROSS_BULL
  - 0.8 × EMA_CROSS_BEAR
```

### Signal Thresholds
- **BUY**: BUY_SCORE ≥ 1.8
- **SELL**: BUY_SCORE ≤ -0.8
- **HOLD**: -0.8 < BUY_SCORE < 1.8

## Risk Levels

Based on predicted volatility (move_pct):
- **LOW RISK**: move_pct < 0.8%
- **MEDIUM RISK**: 0.8% ≤ move_pct ≤ 1.8%
- **HIGH RISK**: move_pct > 1.8%

## Training Data

- **Ticker**: RELIANCE.NS (default)
- **Period**: 5 years
- **Interval**: Daily (1d)
- **Train/Test Split**: 80/20 (time-series split)
- **Sequence Length**: 30 days
- **Total Features**: 13
- **Total Targets**: 10

## Customization

### Change Training Ticker

Edit `train_models.py`:
```python
TICKER = "TCS.NS"  # or any other ticker
```

Then retrain:
```bash
python train_models.py
```

### Adjust EMA Pairs

Edit `train_models.py`:
```python
CANDIDATE_EMA_PAIRS = [
    (10, 20),
    (12, 26),
    (20, 50),
    (8, 21),
]
```

### Change Model Hyperparameters

Edit `train_models.py` in `train_clf()` and `train_reg()` functions:
```python
gru_units = 64
tcn_filters = 64
dense_units = 32
dropout_rate = 0.3
```

## Integration with Backend

The Node.js backend can call this service:

```javascript
const response = await fetch('http://localhost:8001/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({symbol: 'RELIANCE.NS', name: 'Reliance Industries'})
});
const prediction = await response.json();
```

## Troubleshooting

### Models Not Loading
```
Error: "Models not loaded"
Solution: Run python train_models.py first
```

### Memory Issues During Training
- Reduce `batch_size` in `train_models.py`
- Reduce `SEQ_LEN` value
- Use GPU acceleration if available

### Poor Predictions
- Ensure you have at least 1000 days of historical data
- Retrain with different EMA pairs
- Check if ticker is correct (use .NS for NSE, .BO for BSE)

## Performance Notes

- **Training Time**: 10-15 minutes on CPU
- **Prediction Time**: <100ms per stock
- **Memory Usage**: ~500MB for all models
- **GPU Support**: Yes (TensorFlow will use GPU if available)

## References

- **EMA**: Exponential Moving Average
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index
- **ATR**: Average True Range
- **GRU**: Gated Recurrent Unit
- **TCN**: Temporal Convolutional Network
