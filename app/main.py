from fastapi import FastAPI

app = FastAPI(title="Stock ML API")


@app.get("/")
def health_check():
    return {"status": "online"}


@app.get("/signal/{ticker}")
def get_signal(ticker: str):
    return {
        "ticker": ticker.upper(),
        "signal": "hold",
        "confidence": 0.0,
        "message": "Model not yet loaded",
    }
