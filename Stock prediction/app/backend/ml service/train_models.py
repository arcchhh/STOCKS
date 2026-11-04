"""
MTR-QNet Lite - Model Training Script
Trains and saves all models for stock prediction
Run this ONCE to train models, then use model_loader.py for predictions
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import warnings

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# ===========================
# PARAMETERS
# ===========================
TICKER = "RELIANCE.NS"
PERIOD = "5y"
INTERVAL = "1d"
TEST_RATIO = 0.2
RANGE_BINS = [0, 0.005, 0.01, 1]
BREAKOUT_WINDOW = 20
VOLUME_SPIKE_FACTOR = 2.0
RANDOM_STATE = 42
SEQ_LEN = 30
RANGE_MULTIPLIER = 0.8

CANDIDATE_EMA_PAIRS = [
    (10, 20),
    (12, 26),
    (20, 50),
    (8, 21),
]

# Model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================
# HELPER: AUTO-SELECT EMA PAIR
# ===========================
def auto_select_ema_pair(df, candidate_pairs):
    """Auto-select best EMA pair based on heuristic scoring."""
    close = df["Close"]
    best_score = -1
    best_pair = candidate_pairs[0]

    for fast, slow in candidate_pairs:
        if fast >= slow:
            continue

        ema_fast = EMAIndicator(close, fast).ema_indicator()
        ema_slow = EMAIndicator(close, slow).ema_indicator()

        diff = ema_fast - ema_slow
        zone = np.sign(diff)

        valid = zone[zone != 0]
        if len(valid) < 50:
            continue

        bull_frac = (valid == 1).mean()
        bear_frac = (valid == -1).mean()
        imbalance = abs(bull_frac - bear_frac)

        sign_prev = valid.shift(1)
        cross_events = ((valid * sign_prev) == -1).sum()
        cross_rate = cross_events / len(valid)

        score = (1 - imbalance) + min(cross_rate * 10.0, 1.0)

        if score > best_score:
            best_score = score
            best_pair = (fast, slow)

    print(f"[Auto-EMA] Selected: fast={best_pair[0]}, slow={best_pair[1]}, score={best_score:.3f}")
    return best_pair


# ===========================
# BUILD SEQUENCES
# ===========================
def build_seqs_all(X_arr, seq_len):
    """Build sequences of length seq_len for each time index."""
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


# ===========================
# WRAPPER CLASSES
# ===========================
class KerasClassifierWrapper:
    def __init__(self, model, index_to_label, x_mean, x_std, seq_len):
        self.model = model
        self.index_to_label = index_to_label
        self.x_mean = x_mean
        self.x_std = x_std
        self.seq_len = seq_len

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = (X_arr - self.x_mean) / (self.x_std + 1e-8)
        X_seq = build_seqs_all(X_arr, self.seq_len)
        return X_seq

    def predict(self, X):
        X_seq = self._prepare_X(X)
        probs = self.model.predict(X_seq, verbose=0)
        idx = np.argmax(probs, axis=1)
        labels = np.array([self.index_to_label[i] for i in idx])
        return labels


class KerasRegressorWrapper:
    def __init__(self, model, x_mean, x_std, seq_len, y_mean, y_std):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.seq_len = seq_len
        self.y_mean = y_mean
        self.y_std = y_std

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = (X_arr - self.x_mean) / (self.x_std + 1e-8)
        X_seq = build_seqs_all(X_arr, self.seq_len)
        return X_seq

    def predict(self, X):
        X_seq = self._prepare_X(X)
        y_norm = self.model.predict(X_seq, verbose=0).flatten()
        y = y_norm * self.y_std + self.y_mean
        return y


# ===========================
# TRAIN FUNCTIONS
# ===========================
def train_clf(Xt, yt, Xv, yv, name, overfit_protection=False, oversample_minor=False):
    """Train GRU+Attention+TCN hybrid classifier."""
    if oversample_minor:
        tr = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True)], axis=1)
        target_col = tr.columns[-1]
        counts = tr[target_col].value_counts()
        if counts.max() > 0 and len(counts) > 1:
            dfs = []
            max_n = counts.max()
            for cls in counts.index:
                df_cls = tr[tr[target_col] == cls]
                if len(df_cls) == 0:
                    continue
                if len(df_cls) < max_n:
                    df_up = resample(df_cls, replace=True, n_samples=max_n, random_state=RANDOM_STATE)
                else:
                    df_up = df_cls
                dfs.append(df_up)
            tr = pd.concat(dfs, axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            Xt = tr.drop(target_col, axis=1)
            yt = tr[target_col]

    X_train_arr = Xt.to_numpy(dtype=np.float32)
    X_val_arr = Xv.to_numpy(dtype=np.float32)

    x_mean = X_train_arr.mean(axis=0, keepdims=True)
    x_std = X_train_arr.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    X_train_scaled = (X_train_arr - x_mean) / x_std
    X_val_scaled = (X_val_arr - x_mean) / x_std

    X_train_seq = build_seqs_all(X_train_scaled, SEQ_LEN)
    X_val_seq = build_seqs_all(X_val_scaled, SEQ_LEN)

    unique_labels = np.sort(yt.unique())
    label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
    index_to_label = {i: lab for lab, i in label_to_index.items()}

    y_train_idx = yt.map(label_to_index).astype(int).to_numpy()
    y_val_idx = yv.map(label_to_index).astype(int).to_numpy()

    n_classes = len(unique_labels)
    input_dim = X_train_seq.shape[-1]

    if overfit_protection:
        gru_units = 32
        tcn_filters = 32
        dense_units = 16
        dropout_rate = 0.2
    else:
        gru_units = 64
        tcn_filters = 64
        dense_units = 32
        dropout_rate = 0.3

    # Build model
    inputs = keras.Input(shape=(SEQ_LEN, input_dim))

    gru_seq = layers.GRU(gru_units, return_sequences=True)(inputs)
    att_scores = layers.Dense(1, activation="tanh")(gru_seq)
    att_weights = layers.Softmax(axis=1)(att_scores)
    att_context = layers.Multiply()([gru_seq, att_weights])
    att_context = layers.GlobalAveragePooling1D()(att_context)

    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal", dilation_rate=1, activation="relu")(inputs)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal", dilation_rate=2, activation="relu")(tcn)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal", dilation_rate=4, activation="relu")(tcn)
    tcn_out = layers.GlobalAveragePooling1D()(tcn)

    x = layers.Concatenate()([att_context, tcn_out])
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    counts = np.bincount(y_train_idx)
    total = len(y_train_idx)
    class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}

    callback_es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_seq, y_train_idx,
        epochs=80,
        batch_size=32,
        validation_data=(X_val_seq, y_val_idx),
        callbacks=[callback_es],
        class_weight=class_weight,
        verbose=0
    )

    wrapped = KerasClassifierWrapper(model, index_to_label, x_mean, x_std, SEQ_LEN)
    pred_labels = wrapped.predict(Xv)
    pred_series = pd.Series(pred_labels, index=Xv.index)

    print(f"\n-- {name} --")
    print("Accuracy:", round(accuracy_score(yv, pred_series), 4))
    print("F1:", round(f1_score(yv, pred_series, average='weighted'), 4))

    return wrapped, pred_series


def train_reg(Xt, yt, Xv, yv, name):
    """Train GRU+Attention+TCN hybrid regressor."""
    X_train_arr = Xt.to_numpy(dtype=np.float32)
    X_val_arr = Xv.to_numpy(dtype=np.float32)

    x_mean = X_train_arr.mean(axis=0, keepdims=True)
    x_std = X_train_arr.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    X_train_scaled = (X_train_arr - x_mean) / x_std
    X_val_scaled = (X_val_arr - x_mean) / x_std

    X_train_seq = build_seqs_all(X_train_scaled, SEQ_LEN)
    X_val_seq = build_seqs_all(X_val_scaled, SEQ_LEN)

    y_train_arr = yt.to_numpy(dtype=np.float32)
    y_val_arr = yv.to_numpy(dtype=np.float32)

    y_mean = y_train_arr.mean()
    y_std = y_train_arr.std()
    if y_std == 0:
        y_std = 1.0

    y_train_norm = (y_train_arr - y_mean) / y_std
    y_val_norm = (y_val_arr - y_mean) / y_std

    input_dim = X_train_seq.shape[-1]
    gru_units, tcn_filters, dense_units, dropout_rate = 64, 64, 32, 0.3

    inputs = keras.Input(shape=(SEQ_LEN, input_dim))

    gru_seq = layers.GRU(gru_units, return_sequences=True)(inputs)
    att_scores = layers.Dense(1, activation="tanh")(gru_seq)
    att_weights = layers.Softmax(axis=1)(att_scores)
    att_context = layers.Multiply()([gru_seq, att_weights])
    att_context = layers.GlobalAveragePooling1D()(att_context)

    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal", dilation_rate=1, activation="relu")(inputs)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal", dilation_rate=2, activation="relu")(tcn)
    tcn = layers.Conv1D(tcn_filters, kernel_size=3, padding="causal", dilation_rate=4, activation="relu")(tcn)
    tcn_out = layers.GlobalAveragePooling1D()(tcn)

    x = layers.Concatenate()([att_context, tcn_out])
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    callback_es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_seq, y_train_norm,
        epochs=80,
        batch_size=32,
        validation_data=(X_val_seq, y_val_norm),
        callbacks=[callback_es],
        verbose=0
    )

    wrapped = KerasRegressorWrapper(model, x_mean, x_std, SEQ_LEN, y_mean, y_std)
    pred_vals = wrapped.predict(Xv)
    pred_series = pd.Series(pred_vals, index=Xv.index)

    print(f"\n-- {name} --")
    print("R²: ", round(np.dot(pred_series, yv) / (np.linalg.norm(pred_series) * np.linalg.norm(yv) + 1e-8), 4))

    return wrapped, pred_series


# ===========================
# MAIN TRAINING
# ===========================
def train_all_models():
    """Download data and train all models."""
    print("Downloading data...")
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].values.reshape(-1)

    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    print(f"Loaded: {df.shape}")

    # Auto-select EMA
    EMA_FAST, EMA_SLOW = auto_select_ema_pair(df, CANDIDATE_EMA_PAIRS)

    # Calculate indicators
    df["EMA_fast"] = EMAIndicator(df["Close"], EMA_FAST).ema_indicator()
    df["EMA_slow"] = EMAIndicator(df["Close"], EMA_SLOW).ema_indicator()
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

    # Future labels
    df["Range_pct_next"] = ((df["High"] - df["Low"]) / df["Close"]).shift(-1)
    df["Range_Bucket_raw"] = pd.cut(df["Range_pct_next"], bins=RANGE_BINS, labels=[0, 1, 2])

    df["Gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Gap_Label_raw"] = pd.cut(df["Gap_pct"], bins=[-1, -0.005, 0.005, 1], labels=[-1, 0, 1])

    df["Vol_next"] = df["Volume"].shift(-1)
    df["Vol_median20"] = df["Volume"].rolling(20).median()
    df["Vol_Spike_raw"] = (df["Vol_next"] > df["Vol_median20"] * VOLUME_SPIKE_FACTOR).astype(int)

    df["Return_next"] = df["Return"].shift(-1)
    df["Momentum_Continue_raw"] = (np.sign(df["Return"]) == np.sign(df["Return_next"])).astype(int)

    df["20d_high"] = df["High"].rolling(BREAKOUT_WINDOW).max()
    df["Close_next"] = df["Close"].shift(-1)
    df["Breakout_raw"] = (df["Close_next"] > df["20d_high"]).astype(int)

    df["EMA_fast_next"] = df["EMA_fast"].shift(-1)
    df["EMA_slow_next"] = df["EMA_slow"].shift(-1)

    def ema_cross_real(row):
        prev_diff = row["EMA_fast"] - row["EMA_slow"]
        next_diff = row["EMA_fast_next"] - row["EMA_slow_next"]
        if prev_diff <= 0 and next_diff > 0:
            return 1
        if prev_diff >= 0 and next_diff < 0:
            return -1
        return 0

    df["EMA_Crossover_raw"] = df.apply(ema_cross_real, axis=1)

    def ema_zone_next(row):
        if row["EMA_fast_next"] > row["EMA_slow_next"] * 1.001:
            return 1
        if row["EMA_fast_next"] < row["EMA_slow_next"] * 0.999:
            return -1
        return 0

    df["EMA_Trend_Zone_raw"] = df.apply(ema_zone_next, axis=1)

    df["ATR_next"] = df["ATR_14"].shift(-1)

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

    # Final labels
    df["Range_Bucket"] = df["Range_Bucket_raw"].astype("Int64")
    df["Gap_Label"] = df["Gap_Label_raw"].astype("Int64")
    df["Vol_Spike"] = df["Vol_Spike_raw"]
    df["Momentum_Continue"] = df["Momentum_Continue_raw"]
    df["Breakout"] = df["Breakout_raw"]

    df["EMA_Crossover"] = df["EMA_Crossover_raw"].astype(int)
    df["EMA_Trend_Zone"] = df["EMA_Trend_Zone_raw"].astype(int)
    df["ATR_Target"] = df["ATR_next"].astype(float)

    def trend_regime(row):
        if row["Close"] > row["EMA_slow"] * 1.002:
            return 1
        if row["Close"] < row["EMA_slow"] * 0.998:
            return -1
        return 0

    df["Trend_Regime"] = df.apply(trend_regime, axis=1)
    df["Trend_Regime"] = df["Trend_Regime"].rolling(3, min_periods=1).mean()
    df["Trend_Regime"] = df["Trend_Regime"].apply(
        lambda x: 1 if x > 0.33 else (-1 if x < -0.33 else 0)
    )
    df["Trend_Regime"] = df["Trend_Regime"].shift(1)

    df.dropna(inplace=True)

    # X and Y
    X = df[shift_cols]

    y_trend = df["Trend_Regime"]
    y_vol = (df["ATR_14"] > df["ATR_14"].rolling(30).median()).astype(int)
    y_range = df["Range_Bucket"]
    y_gap = (df["Gap_Label"] + 1).astype(int)
    y_volspike = df["Vol_Spike"]
    y_momentum = df["Momentum_Continue"]
    y_breakout = df["Breakout"]
    y_ema_cross = df["EMA_Crossover"]
    y_ema_zone = df["EMA_Trend_Zone"]
    y_atr = df["ATR_Target"]

    # Time-series split
    split = int(len(X) * (1 - TEST_RATIO))
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    def split_y(y):
        return y.iloc[:split], y.iloc[split:]

    y_tr_train, y_tr_test = split_y(y_trend)
    y_vo_train, y_vo_test = split_y(y_vol)
    y_ra_train, y_ra_test = split_y(y_range)
    y_ga_train, y_ga_test = split_y(y_gap)
    y_vs_train, y_vs_test = split_y(y_volspike)
    y_mo_train, y_mo_test = split_y(y_momentum)
    y_br_train, y_br_test = split_y(y_breakout)
    y_ec_train, y_ec_test = split_y(y_ema_cross)
    y_ez_train, y_ez_test = split_y(y_ema_zone)
    y_ar_train, y_ar_test = split_y(y_atr)

    print("\n====== Training Models ======")

    # Train all models
    clf_trend, _ = train_clf(X_train, y_tr_train, X_test, y_tr_test, "Trend Regime")
    clf_vol, _ = train_clf(X_train, y_vo_train, X_test, y_vo_test, "Volatility Regime")
    clf_range, _ = train_clf(X_train, y_ra_train, X_test, y_ra_test, "Range Bucket")
    clf_gap, _ = train_clf(X_train, y_ga_train, X_test, y_ga_test, "Gap Direction")
    clf_vsp, _ = train_clf(X_train, y_vs_train, X_test, y_vs_test, "Volume Spike")
    clf_mom, _ = train_clf(X_train, y_mo_train, X_test, y_mo_test, "Momentum Continue")
    clf_break, _ = train_clf(X_train, y_br_train, X_test, y_br_test, "Breakout")
    clf_ez, _ = train_clf(X_train, y_ez_train, X_test, y_ez_test, "EMA Trend Zone")
    clf_ec, _ = train_clf(X_train, y_ec_train, X_test, y_ec_test, "EMA Crossover", overfit_protection=True, oversample_minor=True)
    reg_atr, _ = train_reg(X_train, y_ar_train, X_test, y_ar_test, "ATR14 Regression")

    # Save models
    print("\n====== Saving Models ======")
    models = {
        "trend": clf_trend,
        "vol": clf_vol,
        "range": clf_range,
        "gap": clf_gap,
        "vsp": clf_vsp,
        "mom": clf_mom,
        "break": clf_break,
        "ez": clf_ez,
        "ec": clf_ec,
        "atr": reg_atr,
    }

    for name, model in models.items():
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {name}")

    # Save metadata
    metadata = {
        "EMA_FAST": EMA_FAST,
        "EMA_SLOW": EMA_SLOW,
        "SEQ_LEN": SEQ_LEN,
        "RANGE_MULTIPLIER": RANGE_MULTIPLIER,
        "ticker": TICKER,
    }

    meta_path = os.path.join(MODEL_DIR, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print("Saved: metadata")

    # Save last row scalings
    X_arr = X.to_numpy(dtype=np.float32)
    x_mean = X_arr.mean(axis=0, keepdims=True)
    x_std = X_arr.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0

    scalings = {
        "x_mean": x_mean,
        "x_std": x_std,
        "last_close": float(df["Close"].iloc[-1]),
        "last_atr": float(df["ATR_14"].iloc[-1]),
    }

    scaling_path = os.path.join(MODEL_DIR, "scalings.pkl")
    with open(scaling_path, "wb") as f:
        pickle.dump(scalings, f)
    print("Saved: scalings")

    print("\n✓ All models trained and saved successfully!")


if __name__ == "__main__":
    train_all_models()
