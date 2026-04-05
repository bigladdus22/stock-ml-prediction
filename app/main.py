from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Stock ML API")

def build_features(df):
    df = df.copy()
    df.columns = df.columns.get_level_values(0)

    # EMAs
    df["ema_10"] = df["Close"].ewm(span=10).mean()
    df["ema_30"] = df["Close"].ewm(span=30).mean()
    df["ema_cross"] = (df["ema_10"] > df["ema_30"]).astype(int)

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # MACD
    df["macd"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Lagged returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)

    # Volatility
    df["volatility"] = df["return_1d"].rolling(10).std()

    # Volume change
    df["volume_change"] = df["Volume"].pct_change(1)

    return df

def create_labels(df):
    # Label based on next day return
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    df["label"] = "hold"
    df.loc[future_return > 0.01, "label"] = "buy"
    df.loc[future_return < -0.01, "label"] = "sell"
    return df

def train_and_predict(ticker: str):
    # Download 2 years of data
    df = yf.download(ticker, period="2y", interval="1d", progress=False)

    df = build_features(df)
    df = create_labels(df)
    df.dropna(inplace=True)

    feature_cols = [
        "ema_10", "ema_30", "ema_cross",
        "rsi", "macd", "macd_signal", "macd_hist",
        "return_1d", "return_3d", "return_5d",
        "volatility", "volume_change"
    ]

    X = df[feature_cols]
    y = df["label"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train on all but last row
    X_train = X.iloc[:-1]
    y_train = y_encoded[:-1]

    # Predict on latest row
    X_latest = X.iloc[[-1]]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_latest)[0]
    proba = model.predict_proba(X_latest)[0]
    confidence = round(float(max(proba)), 2)
    signal = le.inverse_transform([pred])[0]

    latest = df.iloc[-1]
    return {
        "close": round(float(latest["Close"]), 2),
        "rsi": round(float(latest["rsi"]), 2),
        "macd": round(float(latest["macd"]), 4),
        "ema_10": round(float(latest["ema_10"]), 2),
        "ema_30": round(float(latest["ema_30"]), 2),
    }, signal, confidence

@app.get("/")
def health_check():
    return {"status": "online"}

@app.get("/signal/{ticker}")
def get_signal(ticker: str):
    features, signal, confidence = train_and_predict(ticker.upper())
    return {
        "ticker": ticker.upper(),
        "signal": signal,
        "confidence": confidence,
        "features": features
    }