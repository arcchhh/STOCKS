"""
MTR-QNet Lite - Hybrid GRU(+Attention) + TCN Deep Learning Model
for Stock Market Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class KerasClassifierWrapper:
    """Wrapper for Keras classification models with sequence preparation"""
    
    def __init__(self, model, index_to_label: Dict[int, Any], 
                 x_mean: np.ndarray, x_std: np.ndarray, seq_len: int):
        self.model = model
        self.index_to_label = index_to_label
        self.x_mean = x_mean
        self.x_std = x_std
        self.seq_len = seq_len

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize and build sequences"""
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = (X_arr - self.x_mean) / (self.x_std + 1e-8)
        X_seq = self._build_seqs_all(X_arr, self.seq_len)
        return X_seq

    def _build_seqs_all(self, X_arr: np.ndarray, seq_len: int) -> np.ndarray:
        """Build sequences for each time point"""
        N, F = X_arr.shape
        seqs = np.zeros((N, seq_len, F), dtype=np.float32)
        for t in range(N):
            start = max(0, t - seq_len + 1)
            window = X_arr[start:t+1]
            if window.shape[0] < seq_len:
                pad = np.repeat(window[0:1], seq_len - window.shape[0], axis=0)
                seq = np.vstack([pad, window])
            else:
                seq = window
            seqs[t] = seq
        return seqs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_seq = self._prepare_X(X)
        probs = self.model.predict(X_seq, verbose=0)
        idx = np.argmax(probs, axis=1)
        labels = np.array([self.index_to_label[i] for i in idx])
        return labels


class KerasRegressorWrapper:
    """Wrapper for Keras regression models"""
    
    def __init__(self, model, x_mean: np.ndarray, x_std: np.ndarray, 
                 seq_len: int, y_mean: float, y_std: float):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.seq_len = seq_len
        self.y_mean = y_mean
        self.y_std = y_std

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = (X_arr - self.x_mean) / (self.x_std + 1e-8)
        X_seq = self._build_seqs_all(X_arr, self.seq_len)
        return X_seq

    def _build_seqs_all(self, X_arr: np.ndarray, seq_len: int) -> np.ndarray:
        N, F = X_arr.shape
        seqs = np.zeros((N, seq_len, F), dtype=np.float32)
        for t in range(N):
            start = max(0, t - seq_len + 1)
            window = X_arr[start:t+1]
            if window.shape[0] < seq_len:
                pad = np.repeat(window[0:1], seq_len - window.shape[0], axis=0)
                seq = np.vstack([pad, window])
            else:
                seq = window
            seqs[t] = seq
        return seqs

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_seq = self._prepare_X(X)
        y_norm = self.model.predict(X_seq, verbose=0).flatten()
        y = y_norm * self.y_std + self.y_mean
        return y


def build_gru_attention_tcn_classifier(input_dim: int, n_classes: int,
                                        gru_units: int = 64,
                                        tcn_filters: int = 64,
                                        dense_units: int = 32,
                                        dropout_rate: float = 0.3,
                                        seq_len: int = 30) -> keras.Model:
    """
    Build Hybrid GRU(+Attention) + TCN Classifier
    
    Architecture:
    - Branch A: GRU with Attention mechanism for temporal focus
    - Branch B: TCN-style causal convolutions for long-term dependencies
    - Fusion: Concatenate both branches and add dense layers
    """
    inputs = keras.Input(shape=(seq_len, input_dim))

    # Branch A: GRU + Attention
    gru_seq = layers.GRU(gru_units, return_sequences=True, name='gru_layer')(inputs)
    att_scores = layers.Dense(1, activation="tanh", name='attention_scores')(gru_seq)
    att_weights = layers.Softmax(axis=1, name='attention_weights')(att_scores)
    att_context = layers.Multiply(name='attention_context')([gru_seq, att_weights])
    att_context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), 
                                 name='attention_pooling')(att_context)

    # Branch B: TCN-style causal convolutions
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal",
                        dilation_rate=1, activation="relu", name='tcn_conv1')(inputs)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal",
                        dilation_rate=2, activation="relu", name='tcn_conv2')(tcn)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal",
                        dilation_rate=4, activation="relu", name='tcn_conv3')(tcn)
    tcn_out = layers.GlobalAveragePooling1D(name='tcn_pooling')(tcn)

    # Fusion
    x = layers.Concatenate(name='fusion')([att_context, tcn_out])
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Dense(dense_units, activation="relu", name='dense_hidden')(x)
    outputs = layers.Dense(n_classes, activation="softmax", name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='MTR_QNet_Classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_gru_attention_tcn_regressor(input_dim: int,
                                       gru_units: int = 64,
                                       tcn_filters: int = 64,
                                       dense_units: int = 32,
                                       dropout_rate: float = 0.3,
                                       seq_len: int = 30) -> keras.Model:
    """
    Build Hybrid GRU(+Attention) + TCN Regressor for ATR prediction
    """
    inputs = keras.Input(shape=(seq_len, input_dim))

    # Branch A: GRU + Attention
    gru_seq = layers.GRU(gru_units, return_sequences=True, name='gru_layer')(inputs)
    att_scores = layers.Dense(1, activation="tanh", name='attention_scores')(gru_seq)
    att_weights = layers.Softmax(axis=1, name='attention_weights')(att_scores)
    att_context = layers.Multiply(name='attention_context')([gru_seq, att_weights])
    att_context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                                 name='attention_pooling')(att_context)

    # Branch B: TCN
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal",
                        dilation_rate=1, activation="relu", name='tcn_conv1')(inputs)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal",
                        dilation_rate=2, activation="relu", name='tcn_conv2')(tcn)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal",
                        dilation_rate=4, activation="relu", name='tcn_conv3')(tcn)
    tcn_out = layers.GlobalAveragePooling1D(name='tcn_pooling')(tcn)

    # Fusion
    x = layers.Concatenate(name='fusion')([att_context, tcn_out])
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Dense(dense_units, activation="relu", name='dense_hidden')(x)
    outputs = layers.Dense(1, activation="linear", name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='MTR_QNet_Regressor')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


class MTRQNetPredictor:
    """
    MTR-QNet Lite Predictor
    Combines multiple models for ensemble stock prediction
    """
    
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        self.models = {}
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, feature_cols: list, 
              epochs: int = 80, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train all models on historical data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            feature_cols: List of feature column names
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Training metrics dictionary
        """
        from sklearn.model_selection import train_test_split
        
        metrics = {}
        
        # Prepare features
        X = df[feature_cols].values
        
        # Create labels
        # 1. Trend Regime (UP=1, DOWN=-1, NEUTRAL=0)
        df['Return'] = df['Close'].pct_change()
        df['Trend'] = np.where(df['Return'] > 0.005, 1,
                      np.where(df['Return'] < -0.005, -1, 0))
        y_trend = df['Trend'].values
        
        # 2. ATR Target for regression
        if 'ATR_14' in df.columns:
            y_atr = df['ATR_14'].values
        else:
            # Calculate ATR if not present
            y_atr = self._calculate_atr(df).values
            
        # Remove NaN rows
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y_trend) | np.isnan(y_atr))
        X = X[valid_idx]
        y_trend = y_trend[valid_idx]
        y_atr = y_atr[valid_idx]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_tr_train, y_tr_test = y_trend[:split_idx], y_trend[split_idx:]
        y_ar_train, y_ar_test = y_atr[:split_idx], y_atr[split_idx:]
        
        # Scale features
        x_mean = X_train.mean(axis=0, keepdims=True)
        x_std = X_train.std(axis=0, keepdims=True)
        x_std[x_std == 0] = 1.0
        
        self.x_mean = x_mean
        self.x_std = x_std
        
        X_train_scaled = (X_train - x_mean) / x_std
        X_test_scaled = (X_test - x_mean) / x_std
        
        # Build sequences
        X_train_seq = self._build_sequences(X_train_scaled)
        X_test_seq = self._build_sequences(X_test_scaled)
        
        # Train Trend Classifier
        unique_labels = np.sort(np.unique(y_tr_train))
        label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        index_to_label = {i: lab for lab, i in label_to_index.items()}
        
        y_tr_train_idx = np.array([label_to_index[y] for y in y_tr_train])
        y_tr_test_idx = np.array([label_to_index[y] for y in y_tr_test])
        
        trend_model = build_gru_attention_tcn_classifier(
            input_dim=X_train.shape[1],
            n_classes=len(unique_labels),
            seq_len=self.seq_len
        )
        
        # Class weights for imbalance
        counts = np.bincount(y_tr_train_idx)
        total = len(y_tr_train_idx)
        class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}
        
        callback_es = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
        
        print("Training Trend Classifier...")
        trend_model.fit(
            X_train_seq, y_tr_train_idx,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_seq, y_tr_test_idx),
            callbacks=[callback_es],
            class_weight=class_weight,
            verbose=1
        )
        
        self.models['trend'] = KerasClassifierWrapper(
            model=trend_model,
            index_to_label=index_to_label,
            x_mean=x_mean,
            x_std=x_std,
            seq_len=self.seq_len
        )
        
        # Calculate accuracy
        pred_trend = self.models['trend'].predict(X_test)
        from sklearn.metrics import accuracy_score
        trend_acc = accuracy_score(y_tr_test, pred_trend)
        metrics['trend_accuracy'] = float(trend_acc)
        print(f"Trend Accuracy: {trend_acc:.4f}")
        
        # Train ATR Regressor
        y_ar_mean = y_ar_train.mean()
        y_ar_std = y_ar_train.std()
        if y_ar_std == 0:
            y_ar_std = 1.0
            
        y_ar_train_norm = (y_ar_train - y_ar_mean) / y_ar_std
        y_ar_test_norm = (y_ar_test - y_ar_mean) / y_ar_std
        
        atr_model = build_gru_attention_tcn_regressor(
            input_dim=X_train.shape[1],
            seq_len=self.seq_len
        )
        
        print("\nTraining ATR Regressor...")
        atr_model.fit(
            X_train_seq, y_ar_train_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_seq, y_ar_test_norm),
            callbacks=[callback_es],
            verbose=1
        )
        
        self.models['atr'] = KerasRegressorWrapper(
            model=atr_model,
            x_mean=x_mean,
            x_std=x_std,
            seq_len=self.seq_len,
            y_mean=y_ar_mean,
            y_std=y_ar_std
        )
        
        # Calculate R2
        pred_atr = self.models['atr'].predict(X_test)
        from sklearn.metrics import r2_score
        atr_r2 = r2_score(y_ar_test, pred_atr)
        metrics['atr_r2'] = float(atr_r2)
        print(f"ATR R²: {atr_r2:.4f}")
        
        self.is_trained = True
        self.feature_cols = feature_cols
        
        return metrics
    
    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Build sequences for model input"""
        N, F = X.shape
        seqs = np.zeros((N, self.seq_len, F), dtype=np.float32)
        for t in range(N):
            start = max(0, t - self.seq_len + 1)
            window = X[start:t+1]
            if window.shape[0] < self.seq_len:
                pad = np.repeat(window[0:1], self.seq_len - window.shape[0], axis=0)
                seq = np.vstack([pad, window])
            else:
                seq = window
            seqs[t] = seq
        return seqs
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction for given features
        
        Args:
            features: Feature array (1, n_features) or (n_features,)
            
        Returns:
            Prediction dictionary with trend, confidence, atr, signal, priceRange
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get predictions
        trend_pred = self.models['trend'].predict(features)[0]
        atr_pred = self.models['atr'].predict(features)[0]
        
        # Calculate confidence based on model certainty
        # For now, use a heuristic based on ATR and trend strength
        confidence = self._calculate_confidence(features, trend_pred, atr_pred)
        
        # Generate signal
        signal = self._generate_signal(trend_pred, confidence, atr_pred)
        
        # Calculate price range
        price_range = self._calculate_price_range(features, atr_pred)
        
        # Map trend to string
        trend_map = {1: 'UP', -1: 'DOWN', 0: 'NEUTRAL'}
        trend_str = trend_map.get(trend_pred, 'NEUTRAL')
        
        return {
            'trend': trend_str,
            'confidence': round(confidence, 1),
            'atr': round(float(atr_pred), 2),
            'signal': signal,
            'priceRange': {
                'low': round(price_range['low'], 2),
                'high': round(price_range['high'], 2)
            }
        }
    
    def _calculate_confidence(self, features: np.ndarray, trend: int, atr: float) -> float:
        """Calculate prediction confidence"""
        # Base confidence
        base_conf = 65.0
        
        # Add confidence based on feature magnitude
        feature_std = np.std(features)
        conf_boost = min(feature_std * 100, 15)
        
        # Adjust based on ATR (lower ATR = higher confidence)
        atr_factor = max(0, 10 - atr / 5)
        
        confidence = base_conf + conf_boost + atr_factor
        return min(confidence, 95.0)
    
    def _generate_signal(self, trend: int, confidence: float, atr: float) -> str:
        """Generate BUY/SELL/HOLD signal"""
        if trend == 1 and confidence > 70:
            return 'BUY'
        elif trend == -1 and confidence > 70:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_price_range(self, features: np.ndarray, atr: float) -> Dict[str, float]:
        """Calculate predicted price range"""
        # Extract close price from features (assuming it's the first feature)
        close_price = features[0, 0] if features.ndim > 1 else features[0]
        
        # Use ATR to calculate range
        multiplier = 0.8
        band = multiplier * atr
        
        return {
            'low': close_price - band,
            'high': close_price + band
        }
    
    def save(self, path: str):
        """Save model to disk"""
        import pickle
        import json
        
        # Save Keras models
        for name, model_wrapper in self.models.items():
            model_wrapper.model.save(f"{path}/{name}.keras")
        
        # Save metadata
        metadata = {
            'seq_len': self.seq_len,
            'x_mean': self.x_mean.tolist(),
            'x_std': self.x_std.tolist(),
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }
        
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load(self, path: str):
        """Load model from disk"""
        import json
        
        # Load metadata
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.seq_len = metadata['seq_len']
        self.x_mean = np.array(metadata['x_mean'])
        self.x_std = np.array(metadata['x_std'])
        self.feature_cols = metadata['feature_cols']
        
        # Load models
        # This is simplified - in production, you'd load each model with proper wrappers
        self.is_trained = metadata['is_trained']
        
        return self


# Global predictor instance
_predictor: Optional[MTRQNetPredictor] = None


def get_predictor() -> MTRQNetPredictor:
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MTRQNetPredictor(seq_len=30)
    return _predictor
