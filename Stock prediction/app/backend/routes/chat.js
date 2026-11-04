const express = require('express');
const { body, validationResult } = require('express-validator');
const { authMiddleware } = require('../middleware/auth');
const ChatLog = require('../models/ChatLog');

const router = express.Router();

// Validation
const chatValidation = [
  body('message')
    .trim()
    .isLength({ min: 1, max: 1000 })
    .withMessage('Message must be between 1 and 1000 characters')
];

// Predefined responses for common financial questions
const financialResponses = {
  'atr': 'ATR (Average True Range) is a volatility indicator that measures market volatility by decomposing the entire range of an asset price for that period. Higher ATR values indicate higher volatility, which means larger price movements.',
  
  'confidence': 'Our confidence score (0-100%) represents how certain the AI model is about its prediction. Scores above 80% indicate high confidence, while scores below 60% suggest more uncertainty.',
  
  'buy signal': 'A BUY signal indicates that our AI models predict an upward price movement. This is based on trend analysis, momentum indicators, and volatility patterns. Always do your own research before investing.',
  
  'sell signal': 'A SELL signal indicates that our AI models predict a downward price movement. This could be a good time to take profits or cut losses. Consider your overall portfolio strategy.',
  
  'hold signal': 'A HOLD signal suggests that the market is in a consolidation phase or the direction is unclear. It might be best to wait for clearer signals before making a move.',
  
  'gru': 'GRU (Gated Recurrent Unit) is a type of recurrent neural network that processes sequential data. It\'s particularly good at capturing patterns in time series data like stock prices.',
  
  'tcn': 'TCN (Temporal Convolutional Network) uses convolutional layers to capture long-term dependencies in time series data. It\'s effective for understanding historical patterns in stock movements.',
  
  'attention': 'The Attention mechanism helps the model focus on the most relevant parts of the input sequence when making predictions, similar to how humans pay attention to key information.',
  
  'volatility': 'Volatility measures how much the price of an asset fluctuates. High volatility means larger price swings (higher risk/reward), while low volatility indicates more stable prices.',
  
  'risk management': 'Good risk management includes: 1) Never invest more than you can afford to lose, 2) Diversify your portfolio, 3) Use stop-loss orders, 4) Position size based on volatility (ATR).',
};

// Generate AI response
function generateAIResponse(message) {
  const lowerMessage = message.toLowerCase();
  
  // Check for keyword matches
  for (const [keyword, response] of Object.entries(financialResponses)) {
    if (lowerMessage.includes(keyword)) {
      return response;
    }
  }
  
  // Default responses
  const defaultResponses = [
    "Based on current market analysis, I recommend monitoring the trend indicators and ATR levels before making any trading decisions. Our AI models suggest being cautious in the current market environment.",
    
    "That's a great question about the markets! Our hybrid GRU+Attention+TCN model analyzes historical patterns to generate predictions. Would you like me to explain any specific aspect of our prediction methodology?",
    
    "For that query, I'd suggest checking our prediction page with the specific stock symbol. The AI will provide trend analysis, confidence scores, and trading signals based on the latest data.",
    
    "Market analysis involves multiple factors including price trends, volume, volatility (ATR), and momentum. Our AI combines these indicators to generate comprehensive predictions.",
    
    "Remember that all investments carry risk. Our predictions are based on historical patterns and machine learning, but past performance doesn't guarantee future results. Always do your own research!",
  ];
  
  // Return random default response
  return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
}

// @route   POST /api/chat
// @desc    Send message to AI chatbot
// @access  Private
router.post('/', authMiddleware, chatValidation, async (req, res) => {
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

    const { message } = req.body;

    // Generate AI response
    const answer = generateAIResponse(message);

    // Save chat log
    const chatLog = new ChatLog({
      userId: req.userId,
      question: message,
      answer: answer
    });

    await chatLog.save();

    res.json({
      success: true,
      data: { answer }
    });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to process chat message'
    });
  }
});

// @route   GET /api/chat/history
// @desc    Get user's chat history
// @access  Private
router.get('/history', authMiddleware, async (req, res) => {
  try {
    const chatLogs = await ChatLog.find({ userId: req.userId })
      .sort({ timestamp: -1 })
      .limit(50);

    res.json({
      success: true,
      count: chatLogs.length,
      data: chatLogs
    });
  } catch (error) {
    console.error('Get chat history error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get chat history'
    });
  }
});

module.exports = router;
