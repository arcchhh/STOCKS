const express = require('express');
const axios = require('axios');
const { body, validationResult } = require('express-validator');
const { authMiddleware } = require('../middleware/auth');
const Prediction = require('../models/Prediction');

const router = express.Router();

// ML Service URL
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';

// Validation
const predictValidation = [
  body('symbol')
    .trim()
    .isLength({ min: 1, max: 20 })
    .withMessage('Stock symbol is required')
    .matches(/^[A-Za-z0-9.]+$/)
    .withMessage('Invalid stock symbol format')
];

// @route   POST /api/predict
// @desc    Get stock prediction from ML service
// @access  Private
router.post('/', authMiddleware, predictValidation, async (req, res) => {
  try {
    // Check validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        message: 'Validation failed',
        errors: errors.array()
      });
    }

    const { symbol } = req.body;
    const upperSymbol = symbol.toUpperCase();

    // Call ML service
    let mlResponse;
    try {
      mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, {
        symbol: upperSymbol
      }, {
        timeout: 30000 // 30 second timeout
      });
    } catch (mlError) {
      console.error('ML service error:', mlError.message);
      
      // Generate mock prediction for demo
      mlResponse = {
        data: generateMockPrediction(upperSymbol)
      };
    }

    const predictionData = mlResponse.data;

    // Save prediction to database
    const prediction = new Prediction({
      userId: req.userId,
      symbol: upperSymbol,
      trend: predictionData.trend,
      confidence: predictionData.confidence,
      atr: predictionData.atr,
      signal: predictionData.signal,
      priceRange: predictionData.priceRange
    });

    await prediction.save();

    res.json({
      success: true,
      data: predictionData
    });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get prediction'
    });
  }
});

// @route   GET /api/predict/history
// @desc    Get user's prediction history
// @access  Private
router.get('/history', authMiddleware, async (req, res) => {
  try {
    const predictions = await Prediction.find({ userId: req.userId })
      .sort({ timestamp: -1 })
      .limit(50);

    res.json({
      success: true,
      count: predictions.length,
      data: predictions
    });
  } catch (error) {
    console.error('Get history error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get prediction history'
    });
  }
});

// Mock prediction generator for demo
function generateMockPrediction(symbol) {
  const trends = ['UP', 'DOWN', 'NEUTRAL'];
  const signals = ['BUY', 'SELL', 'HOLD'];
  
  // Generate deterministic mock based on symbol
  const symbolHash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const trendIndex = symbolHash % 3;
  const signalIndex = (symbolHash + 1) % 3;
  
  const basePrice = 100 + (symbolHash % 4000);
  const volatility = 5 + (symbolHash % 50);
  
  return {
    trend: trends[trendIndex],
    confidence: 60 + (symbolHash % 35),
    atr: volatility,
    signal: signals[signalIndex],
    priceRange: {
      low: basePrice - volatility,
      high: basePrice + volatility
    }
  };
}

module.exports = router;
