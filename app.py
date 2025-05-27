import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="BTC Price Prediction")

API_KEY = "adb2e003-6773-4184-94f4-6578bae25354"
HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY
}
BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"

def fetch_btc_data():
    params = {
        "symbol": "BTC",
        "time_start": (pd.Timestamp.utcnow() - pd.Timedelta(hours=24)).isoformat(),
        "interval": "hourly"
    }
    response = requests.get(BASE_URL, headers=HEADERS, params=params)
    data = response.json()

    if "data" not in data or "quotes" not in data["data"]:
        raise KeyError("Invalid response from CoinMarketCap")

    quotes = data["data"]["quotes"]
    df = pd.DataFrame([{"timestamp": q["timestamp"], "price": q["quote"]["USD"]["price"]} for q in quotes])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def predict_price(df):
    df = df.copy()
    df["time"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["time"]], df["price"])
    future_time = [[len(df) + 3]]
    prediction = model.predict(future_time)
    return prediction[0]

def plot_prices(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    df["price"].plot(ax=ax, title="BTC Price - Last 24h", color="orange")
    ax.set_ylabel("USD")
    ax.set_xlabel("Time")
    ax.grid(True)
    buf = io.BytesIO()
    pl
