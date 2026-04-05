from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = FastAPI(title="Stock ML API")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

def build_features(df):
    df = df.copy()
    df.columns = df.columns.get_level_values(0)
    df["ema_10"] = df["Close"].ewm(span=10).mean()
    df["ema_30"] = df["Close"].ewm(span=30).mean()
    df["ema_cross"] = (df["ema_10"] > df["ema_30"]).astype(int)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    df["macd"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["volatility"] = df["return_1d"].rolling(10).std()
    df["volume_change"] = df["Volume"].pct_change(1)
    return df

def create_labels(df):
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    df["label"] = "hold"
    df.loc[future_return > 0.01, "label"] = "buy"
    df.loc[future_return < -0.01, "label"] = "sell"
    return df

def train_and_predict(ticker: str):
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
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train = X.iloc[:-1]
    y_train = y_encoded[:-1]
    X_latest = X.iloc[[-1]]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
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

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signal/{ticker}")
def get_signal(ticker: str):
    features, signal, confidence = train_and_predict(ticker.upper())
    return {
        "ticker": ticker.upper(),
        "signal": signal,
        "confidence": confidence,
        "features": features
    }
