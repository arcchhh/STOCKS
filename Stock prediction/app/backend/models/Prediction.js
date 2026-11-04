const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  symbol: {
    type: String,
    required: [true, 'Stock symbol is required'],
    uppercase: true,
    trim: true
  },
  trend: {
    type: String,
    enum: ['UP', 'DOWN', 'NEUTRAL'],
    required: true
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  atr: {
    type: Number,
    required: true,
    min: 0
  },
  signal: {
    type: String,
    enum: ['BUY', 'SELL', 'HOLD'],
    required: true
  },
  priceRange: {
    low: {
      type: Number,
      required: true
    },
    high: {
      type: Number,
      required: true
    }
  },
  timestamp: {
    type: Date,
    default: Date.now,
    index: true
  }
});

// Index for faster queries
predictionSchema.index({ userId: 1, timestamp: -1 });

module.exports = mongoose.model('Prediction', predictionSchema);
