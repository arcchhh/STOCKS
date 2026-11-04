const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const Prediction = require('../models/Prediction');
const ChatLog = require('../models/ChatLog');

const router = express.Router();

// @route   GET /api/dashboard/stats
// @desc    Get dashboard statistics
// @access  Private
router.get('/stats', authMiddleware, async (req, res) => {
  try {
    const userId = req.userId;

    // Get prediction counts
    const totalPredictions = await Prediction.countDocuments({ userId });
    const buySignals = await Prediction.countDocuments({ userId, signal: 'BUY' });
    const sellSignals = await Prediction.countDocuments({ userId, signal: 'SELL' });
    const holdSignals = await Prediction.countDocuments({ userId, signal: 'HOLD' });

    // Get recent predictions
    const recentPredictions = await Prediction.find({ userId })
      .sort({ timestamp: -1 })
      .limit(10);

    // Get recent chat logs
    const recentChats = await ChatLog.find({ userId })
      .sort({ timestamp: -1 })
      .limit(5);

    res.json({
      success: true,
      data: {
        totalPredictions,
        buySignals,
        sellSignals,
        holdSignals,
        recentPredictions,
        recentChats
      }
    });
  } catch (error) {
    console.error('Dashboard stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get dashboard statistics'
    });
  }
});

// @route   GET /api/dashboard/export
// @desc    Export predictions to CSV
// @access  Private
router.get('/export', authMiddleware, async (req, res) => {
  try {
    const userId = req.userId;

    // Get all predictions for user
    const predictions = await Prediction.find({ userId })
      .sort({ timestamp: -1 });

    // Create CSV content
    const headers = ['Symbol', 'Trend', 'Confidence', 'ATR', 'Signal', 'Price Low', 'Price High', 'Date'];
    const rows = predictions.map(p => [
      p.symbol,
      p.trend,
      p.confidence,
      p.atr,
      p.signal,
      p.priceRange.low,
      p.priceRange.high,
      p.timestamp.toISOString()
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    // Set headers for file download
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=predictions_${new Date().toISOString().split('T')[0]}.csv`);
    
    res.send(csvContent);
  } catch (error) {
    console.error('Export error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to export data'
    });
  }
});

module.exports = router;
