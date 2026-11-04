"""
Training script for MTR-QNet Lite Model
Run this to train the model on historical stock data
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from model import MTRQNetPredictor

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def fetch_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance"""
    print(f"Fetching data for {symbol} (period: {period})...")
    
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Downloaded {len(df)} rows")
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    print("Calculating technical indicators...")
    
    # EMAs
    df["EMA_fast"] = EMAIndicator(df["Close"], 12).ema_indicator()
    df["EMA_slow"] = EMAIndicator(df["Close"], 26).ema_indicator()
    df["EMA_diff"] = df["EMA_fast"] - df["EMA_slow"]
    
    # MACD
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    # RSI
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(df["Close"], 20, 2)
    df["BB_Width"] = bb.bollinger_hband() - bb.bollinger_lband()
    
    # ATR
    df["ATR_14"] = AverageTrueRange(
        df["High"], df["Low"], df["Close"], 14
    ).average_true_range()
    
    # Lag features
    df["Close_Lag1"] = df["Close"].shift(1)
    df["Return"] = df["Close"].pct_change()
    
    # Volume features
    df["Vol_median20"] = df["Volume"].rolling(20).median()
    
    # Rolling high
    df["20d_high"] = df["High"].rolling(20).max()
    
    print("Features calculated:", [
        "EMA_fast", "EMA_slow", "EMA_diff",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI", "BB_Width", "ATR_14",
        "Close_Lag1", "Return", "Vol_median20", "20d_high"
    ])
    
    return df


def train_model(symbol: str, period: str = "2y", epochs: int = 50, 
                model_dir: str = "./models") -> dict:
    """
    Train the MTR-QNet model
    
    Args:
        symbol: Stock symbol to train on (e.g., "RELIANCE.NS")
        period: Historical data period (e.g., "2y", "5y")
        epochs: Number of training epochs
        model_dir: Directory to save trained model
        
    Returns:
        Training metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"MTR-QNet Lite Training")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {period}")
    print(f"Epochs: {epochs}")
    print(f"Model Dir: {model_dir}")
    print(f"{'='*60}\n")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Fetch data
    df = fetch_data(symbol, period)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows. Need at least 100.")
    
    # Calculate features
    df = calculate_features(df)
    
    # Define feature columns
    feature_cols = [
        "EMA_fast", "EMA_slow", "EMA_diff",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI", "BB_Width", "ATR_14",
        "Close_Lag1", "Return", "Vol_median20", "20d_high"
    ]
    
    # Create and train predictor
    predictor = MTRQNetPredictor(seq_len=30)
    
    metrics = predictor.train(
        df=df,
        feature_cols=feature_cols,
        epochs=epochs,
        batch_size=32
    )
    
    # Save model
    print(f"\nSaving model to {model_dir}...")
    predictor.save(model_dir)
    
    # Save training info
    training_info = {
        "symbol": symbol,
        "period": period,
        "epochs": epochs,
        "data_rows": len(df),
        "metrics": metrics,
        "trained_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(model_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Trend Accuracy: {metrics['trend_accuracy']:.4f}")
    print(f"ATR R²: {metrics['atr_r2']:.4f}")
    print(f"Model saved to: {model_dir}")
    print(f"{'='*60}\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MTR-QNet Lite Model")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS",
                        help="Stock symbol to train on (default: RELIANCE.NS)")
    parser.add_argument("--period", type=str, default="2y",
                        help="Historical data period (default: 2y)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directory to save model (default: ./models)")
    
    args = parser.parse_args()
    
    try:
        metrics = train_model(
            symbol=args.symbol,
            period=args.period,
            epochs=args.epochs,
            model_dir=args.model_dir
        )
        
        print("Training completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
