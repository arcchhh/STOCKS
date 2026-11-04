"""
StockAI ML Service - FastAPI Application
MTR-QNet Lite: Hybrid EMA + Deep Learning Model
Provides real AI-powered stock prediction endpoints
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader import model_manager

# Initialize FastAPI app
app = FastAPI(
    title="StockAI ML Service",
    description="AI-powered stock market prediction API (MTR-QNet Lite)",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    symbol: str
    name: Optional[str] = None

class PredictionResponse(BaseModel):
    symbol: str
    trend: str
    confidence: float
    atr: float
    signal: str
    priceRange: Dict[str, float]
    volatility: Optional[Dict[str, Any]] = None
    indicators: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    message: str

# Check if models exist
models_loaded = model_manager.models_available()


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "StockAI ML Service",
        "version": "2.0.0",
        "status": "running",
        "model": "MTR-QNet Lite (GRU+Attention+TCN Hybrid)",
        "models_loaded": str(models_loaded)
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    status = "healthy" if models_loaded else "degraded"
    message = "All models loaded and ready" if models_loaded else "Models not loaded - using training mode"
    
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        version="2.0.0",
        message=message
    )
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate stock market prediction"""

    symbol = request.symbol.upper().strip()
    name = request.name

    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    if len(symbol) > 20:
        raise HTTPException(status_code=400, detail="Symbol too long")

    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run train_models.py first. Use: python train_models.py"
        )

    prediction = model_manager.predict(symbol, name)

    if prediction is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not generate prediction for symbol: {symbol}"
        )

    clean_prediction = {
        "symbol": str(prediction["symbol"]),
        "trend": str(prediction["trend"]),
        "confidence": float(prediction["confidence"]),
        "atr": float(prediction["atr"]),
        "signal": str(prediction["signal"]),
        "priceRange": {
            "low": float(prediction["priceRange"]["low"]),
            "high": float(prediction["priceRange"]["high"])
        },
        "volatility": None,
        "indicators": None
    }

    return PredictionResponse(**clean_prediction)


          	
@app.get("/models/status")
async def model_status():
    """Get status of loaded models"""
    return {
        "models_loaded": models_loaded,
        "models": list(model_manager.models.keys()),
        "metadata": model_manager.metadata,
        "model_count": len(model_manager.models),
        "required_models": 10,
        "message": "All models ready" if models_loaded else "Models not loaded. Run: python train_models.py"
    }


@app.post("/predict/batch")
async def predict_batch(symbols: list):
    """Batch prediction for multiple symbols"""
    
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run train_models.py first"
        )

    results = []
    for sym in symbols:
        if len(sym) > 20:
            continue
        
        prediction = model_manager.predict(sym.upper())
        if prediction:
            results.append(prediction)
    
    return {"predictions": results, "count": len(results)}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    status_text = "✓ Models Loaded" if models_loaded else "⚠ Models Not Loaded"
    
    print("""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║          StockAI ML Service - MTR-QNet Lite           ║
║                                                       ║
║   Status: {}                           ║
║   Port: 8001                                          ║
║   Model: GRU+Attention+TCN Hybrid                     ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝

Note: If models are not loaded, run: python train_models.py

    """.format(status_text))

    # ================= CHAT =================

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Simple demo response (you can replace with real LLM later)
    return {
        "answer": f"You asked: {req.message}. This is a demo AI response from StockAI."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
