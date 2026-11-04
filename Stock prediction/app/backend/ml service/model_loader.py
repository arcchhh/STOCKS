"""
Model Loader - Load trained models and generate predictions
"""

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings

import __main__
from train_models import KerasClassifierWrapper, KerasRegressorWrapper

__main__.KerasClassifierWrapper = KerasClassifierWrapper
__main__.KerasRegressorWrapper = KerasRegressorWrapper

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class ModelManager:
    def __init__(self):
        """Load all models and metadata on initialization."""
        self.models = {}
        self.metadata = {}
        self.scalings = {}
        self.load_models()

    def load_models(self):
        """Load all saved models and metadata."""
        try:
            # Load models
            model_files = ["trend", "vol", "range", "gap", "vsp", "mom", "break", "ez", "ec", "atr"]
            for model_name in model_files:
                model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        self.models[model_name] = pickle.load(f)
            
            # Load metadata
            meta_path = os.path.join(MODEL_DIR, "metadata.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
            
            # Load scalings
            scaling_path = os.path.join(MODEL_DIR, "scalings.pkl")
            if os.path.exists(scaling_path):
                with open(scaling_path, "rb") as f:
                    self.scalings = pickle.load(f)
            
            print(f"✓ Loaded {len(self.models)} models")
            return len(self.models) > 0
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False

    def models_available(self):
        """Check if all required models are loaded."""
        required = ["trend", "vol", "range", "gap", "vsp", "mom", "break", "ez", "ec", "atr"]
        return all(m in self.models for m in required)

    def get_features(self, ticker, period="5y", interval="1d"):
        """Download and calculate features for a ticker."""
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(inplace=True)
            df.sort_index(inplace=True)

            # Get EMA periods from metadata
            ema_fast = self.metadata.get("EMA_FAST", 10)
            ema_slow = self.metadata.get("EMA_SLOW", 20)

            # Calculate indicators
            df["EMA_fast"] = EMAIndicator(df["Close"], ema_fast).ema_indicator()
            df["EMA_slow"] = EMAIndicator(df["Close"], ema_slow).ema_indicator()
            df["EMA_diff"] = df["EMA_fast"] - df["EMA_slow"]

            macd = MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Hist"] = macd.macd_diff()

            df["RSI"] = RSIIndicator(df["Close"], 14).rsi()

            bb = BollingerBands(df["Close"], 20, 2)
            df["BB_Width"] = bb.bollinger_hband() - bb.bollinger_lband()

            df["ATR_14"] = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()

            df["Close_Lag1"] = df["Close"].shift(1)
            df["Return"] = df["Close"].pct_change()
            df["Vol_median20"] = df["Volume"].rolling(20).median()
            df["20d_high"] = df["High"].rolling(20).max()

            # Shift features
            shift_cols = [
                "EMA_fast", "EMA_slow", "EMA_diff",
                "MACD", "MACD_Signal", "MACD_Hist",
                "RSI", "BB_Width", "ATR_14",
                "Close_Lag1", "Return", "Vol_median20", "20d_high"
            ]

            for col in shift_cols:
                df[col] = df[col].shift(1)

            df.dropna(inplace=True)

            return df, shift_cols

        except Exception as e:
            print(f"✗ Error downloading data: {e}")
            return None, None

    def predict(self, ticker, symbol_name=None):
        """Generate predictions for a ticker."""
        if not self.models_available():
            return None

        # Get features
        df, shift_cols = self.get_features(ticker)
        if df is None or len(df) == 0:
            return None

        X = df[shift_cols]
        last_row = X.tail(1)

        try:
            # Get predictions
            pred_tr = self.models["trend"].predict(last_row)[0]
            pred_mo = self.models["mom"].predict(last_row)[0]
            pred_vo = self.models["vol"].predict(last_row)[0]
            pred_ra = self.models["range"].predict(last_row)[0]
            pred_ga = self.models["gap"].predict(last_row)[0]
            pred_vs = self.models["vsp"].predict(last_row)[0]
            pred_br = self.models["break"].predict(last_row)[0]
            pred_ez = self.models["ez"].predict(last_row)[0]
            pred_ec = self.models["ec"].predict(last_row)[0]
            pred_atr_live = self.models["atr"].predict(last_row)[0]

            # Build signals
            trend_up = int(pred_tr == 1)
            momentum_up = int(pred_mo == 1)
            vol_low = int(pred_vo == 0)
            range_big = int(pred_ra == 2)
            gap_up = int(pred_ga == 2)
            gap_down = int(pred_ga == 0)
            breakout_sig = int(pred_br)
            volume_spike = int(pred_vs)

            ema_zone_bull = int(pred_ez == 1)
            ema_zone_bear = int(pred_ez == -1)
            ema_cross_bull = int(pred_ec == 1)
            ema_cross_bear = int(pred_ec == -1)

            # ATR normalization
            atr_series = pd.Series(
                self.models["atr"].predict(X),
                index=X.index
            )
            atr_min = atr_series.min()
            atr_max = atr_series.max()
            atr_range = atr_max - atr_min if atr_max > atr_min else 1.0
            var_norm_live = (pred_atr_live - atr_min) / (atr_range + 1e-9)
            var_norm_live = max(min(var_norm_live, 1), 0)

            # Compute buy score
            buy_score = (
                + 1.3 * trend_up
                + 0.9 * momentum_up
                + 0.6 * vol_low
                + 0.7 * range_big
                + 1.0 * breakout_sig
                + 0.7 * gap_up
                - 0.4 * gap_down
                - 0.9 * volume_spike
                - var_norm_live
                + 0.35 * ema_zone_bull
                - 0.35 * ema_zone_bear
                + 0.8 * ema_cross_bull
                - 0.8 * ema_cross_bear
            )

            # Generate signal
            if buy_score >= 1.8:
                signal = "BUY"
            elif buy_score <= -0.8:
                signal = "SELL"
            else:
                signal = "HOLD"

            # Calculate price range
            last_close = float(df["Close"].iloc[-1])
            range_multiplier = self.metadata.get("RANGE_MULTIPLIER", 0.8)
            
            # Ensure ATR is valid
            if pred_atr_live <= 0 or np.isnan(pred_atr_live):
                pred_atr_live = float(df["ATR_14"].iloc[-1])
            
            band = range_multiplier * pred_atr_live
            lower_range = last_close - band
            upper_range = last_close + band
            move_pct = (band / last_close * 100) if last_close != 0 else 0

            # Confidence score
            confidence = min(abs(buy_score) * 20 + 50, 95)

            return {
                "symbol": symbol_name or ticker,
                "ticker": ticker,
                "trend": "UP" if pred_tr == 1 else ("DOWN" if pred_tr == -1 else "NEUTRAL"),
                "confidence": round(confidence, 2),
                "atr": round(pred_atr_live, 2),
                "signal": signal,
                "buy_score": round(buy_score, 2),
                "priceRange": {
                    "low": round(lower_range, 2),
                    "high": round(upper_range, 2),
                    "current": round(last_close, 2),
                },
                "volatility": {
                    "move_pct": round(move_pct, 2),
                    "risk_level": "LOW" if move_pct < 0.8 else ("MEDIUM" if move_pct <= 1.8 else "HIGH")
                },
                "indicators": {
                    "trend_regime": pred_tr,
                    "ema_zone": pred_ez,
                    "ema_cross": pred_ec,
                    "momentum": pred_mo,
                    "breakout": pred_br,
                    "gap": pred_ga,
                }
            }

        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None


# Initialize global manager
model_manager = ModelManager()
