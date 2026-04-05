from fastapi import FastAPI
import yfinance as yf
import pandas as pd

app = FastAPI(title="Stock ML API")

def calculate_features(ticker: str):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    df.columns = df.columns.get_level_values(0)
    
    # EMA features
    df["ema_10"] = df["Close"].ewm(span=10).mean()
    df["ema_30"] = df["Close"].ewm(span=30).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    
    # Return latest row
    latest = df.iloc[-1]
    return {
        "close": round(float(latest["Close"]), 2),
        "ema_10": round(float(latest["ema_10"]), 2),
        "ema_30": round(float(latest["ema_30"]), 2),
        "rsi": round(float(latest["rsi"]), 2),
    }

@app.get("/")
def health_check():
    return {"status": "online"}

@app.get("/signal/{ticker}")
def get_signal(ticker: str):
    features = calculate_features(ticker.upper())
    
    # Simple rule-based signal for now (ML model comes next)
    rsi = features["rsi"]
    ema_cross = features["ema_10"] > features["ema_30"]
    
    if rsi < 35 and ema_cross:
        signal = "buy"
    elif rsi > 65 and not ema_cross:
        signal = "sell"
    else:
        signal = "hold"
    
    return {
        "ticker": ticker.upper(),
        "signal": signal,
        "features": features
    }