import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_cache = {}

def add_indicators(df):
    df = df.copy()
    
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, 0.0001)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma + (std * 2)
    df["BB_Lower"] = sma - (std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma
    
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    
    df["VWAP"] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df["Stoch_K"] = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, 1)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    
    df["ROC"] = ((df["Close"] - df["Close"].shift(12)) / df["Close"].shift(12).replace(0, 1)) * 100
    
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, 1)
    
    df["Price_Change"] = df["Close"].pct_change()
    df["Price_Change_5"] = df["Close"].pct_change(5)
    
    return df

def create_features(df, horizon=1):
    df = df.copy()
    df["Target"] = df["Close"].shift(-horizon)
    df["Target_Direction"] = (df["Target"] > df["Close"]).astype(int)
    return df

def get_feature_columns():
    return [
        "Open", "High", "Low", "Volume", "Close",
        "MA20", "MA50", "MA200",
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "BB_Width",
        "ATR", "VWAP", "Stoch_K", "Stoch_D",
        "ROC", "Volume_MA20", "Volume_Ratio",
        "Price_Change", "Price_Change_5"
    ]

def train_model(symbol, horizon=1):
    try:
        df = yf.Ticker(symbol).history(period="2y")
        if df.empty:
            return None, None, None

        df = add_indicators(df)
        df = create_features(df, horizon)
        
        df["MA200"] = df["Close"].rolling(200).mean()
        df = df.dropna()

        min_required = max(30, horizon + 10)
        if len(df) < min_required:
            return None, None, None

        feature_cols = get_feature_columns()
        missing_cols = [c for c in feature_cols if c not in df.columns]
        for c in missing_cols:
            df[c] = 0
        
        X = df[feature_cols].fillna(0)
        y = df["Target"]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        train_pred = model.predict(X)
        train_mae = np.mean(np.abs(y - train_pred))
        y_range = y.max() - y.min()
        train_mape = (train_mae / y_range) * 100 if y_range > 0 else 100

        confidence = max(0, min(100, 100 - train_mape))
        
        return model, df, confidence
    except Exception as e:
        logger.error(f"Model error {symbol}: {e}")
        return None, None, None

def predict(symbol, sentiment=0, horizon=1, use_cache=True):
    try:
        cache_key = f"{symbol}_{horizon}"
        
        if use_cache and cache_key in _model_cache:
            cached_model, cached_df, cached_time, cached_conf = _model_cache[cache_key]
            if sentiment == 0 and horizon == 1 and time.time() - cached_time < 300:
                model, df, confidence = cached_model, cached_df, cached_conf
            else:
                model, df, confidence = train_model(symbol, horizon)
                if model:
                    _model_cache[cache_key] = (model, df, time.time(), confidence)
        else:
            model, df, confidence = train_model(symbol, horizon)
            if model:
                _model_cache[cache_key] = (model, df, time.time(), confidence)

        if model is None or df is None:
            return 0, 0, 0

        feature_cols = get_feature_columns()
        missing_cols = [c for c in feature_cols if c not in df.columns]
        for c in missing_cols:
            df[c] = 0
            
        latest = df.iloc[-1][feature_cols].fillna(0)
        latest_df = latest.to_frame().T

        if sentiment != 0:
            sentiment_adjustment = 1 + (sentiment * 0.02)
            latest_df["Price_Change"] *= sentiment_adjustment
            latest_df["Price_Change_5"] *= sentiment_adjustment

        prediction = model.predict(latest_df)[0]
        current_price = df.iloc[-1]["Close"]

        prediction = max(current_price * 0.5, min(current_price * 2, prediction))

        return current_price, prediction, confidence
    except Exception as e:
        logger.error(f"Predict error {symbol}: {e}")
        return 0, 0, 0