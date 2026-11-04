# StockAI - AI Stock Market Prediction Platform

A full-stack AI-powered stock market prediction web application with a React frontend, Node.js backend, MongoDB database, and Python FastAPI ML service.

![StockAI](https://img.shields.io/badge/StockAI-v1.0.0-blue)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![Node.js](https://img.shields.io/badge/Node.js-20-339933?logo=node.js)
![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python)

## Features

- **AI-Powered Predictions**: GRU + Attention + TCN hybrid model for stock trend forecasting
- **Volatility Analysis**: ATR (Average True Range) calculations for risk management
- **Trading Signals**: Clear BUY/SELL/HOLD recommendations with confidence scores
- **Financial Chatbot**: AI assistant for market analysis and trading advice
- **Dashboard**: Track prediction history with interactive charts
- **Export Data**: Download predictions as CSV
- **Responsive Design**: Dark fintech theme with glassmorphism UI

## Tech Stack

### Frontend
- React 18 + TypeScript
- Tailwind CSS
- Chart.js for data visualization
- Framer Motion for animations
- React Router for navigation

### Backend
- Node.js + Express
- MongoDB + Mongoose
- JWT Authentication
- bcryptjs for password hashing

### ML Service
- Python FastAPI
- TensorFlow (model loading stub)
- NumPy & Pandas for data processing

## Project Structure

```
stockai/
├── src/                      # React Frontend
│   ├── components/           # Reusable components
│   ├── context/              # React contexts (Auth)
│   ├── pages/                # Page components
│   ├── services/             # API services
│   ├── types/                # TypeScript types
│   └── ...
├── backend/                  # Node.js Backend
│   ├── models/               # MongoDB models
│   ├── routes/               # API routes
│   ├── middleware/           # Auth middleware
│   └── server.js             # Entry point
├── ml-service/               # Python FastAPI ML Service
│   ├── main.py               # FastAPI application
│   └── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.9+
- MongoDB (optional - mock mode works without it)

### 1. Clone and Install

```bash
# Frontend dependencies
cd stockai
npm install

# Backend dependencies
cd backend
npm install

# ML Service dependencies
cd ../ml-service
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment files
cp .env.example .env
cp backend/.env.example backend/.env

# Edit the .env files with your configuration
```

### 3. Start Services

**Terminal 1 - ML Service:**
```bash
cd ml-service
python main.py
# or
uvicorn main:app --reload --port 8001
```

**Terminal 2 - Backend:**
```bash
cd backend
npm run dev
# or
npm start
```

**Terminal 3 - Frontend:**
```bash
cd stockai
npm run dev
```

### 4. Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:5000
- ML Service: http://localhost:8001

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user

### Predictions
- `POST /api/predict` - Get stock prediction
- `GET /api/predict/history` - Get prediction history

### Chatbot
- `POST /api/chat` - Send message to AI
- `GET /api/chat/history` - Get chat history

### Dashboard
- `GET /api/dashboard/stats` - Get user statistics
- `GET /api/dashboard/export` - Export predictions as CSV

## ML Model Integration

The ML service is designed to load pre-trained TensorFlow models:

```python
# models/
#   ├── trend.keras      # Trend prediction model
#   ├── momentum.keras   # Momentum analysis model
#   ├── vol.keras        # Volatility prediction
#   ├── atr.keras        # ATR calculation model
#   └── meta.pkl         # Metadata/scaler
```

When models are not available, the service falls back to mock predictions for demonstration.

## Screenshots

### Home Page
Hero section with animated background and feature highlights.

### Prediction Page
Input stock symbols and receive AI-powered predictions with confidence scores.

### Dashboard
View prediction history, signal distribution charts, and export data.

### Chatbot
AI financial assistant for market analysis and trading advice.

## Environment Variables

### Frontend (.env)
```
VITE_API_URL=http://localhost:5000/api
```

### Backend (.env)
```
PORT=5000
MONGODB_URI=mongodb://localhost:27017/stockai
JWT_SECRET=your-secret-key
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
ML_SERVICE_URL=http://localhost:8001
```

## Production Deployment

1. Set `NODE_ENV=production`
2. Use strong JWT_SECRET
3. Configure MongoDB with authentication
4. Set up HTTPS/SSL
5. Use environment-specific API URLs

## License

MIT License - Not financial advice. Use at your own risk.

## Disclaimer

This application is for educational and informational purposes only. Stock predictions are not guaranteed and should not be considered as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
