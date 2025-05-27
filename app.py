import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="BTC Price Predictor", layout="centered")
st.title("Bitcoin Price Prediction")

@st.cache_data
def fetch_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "hourly"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")

    data = response.json()
    if "prices" not in data:
        raise Exception("Missing 'prices' in API response")

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
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
    df["price"].plot(ax=ax, title="BTC Price (Last 24 hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    st.pyplot(fig)

try:
    df = fetch_btc_data()
    prediction = predict_price(df)
    st.subheader(f"${prediction:,.2f}")
    plot_prices(df)
except Exception as e:
    st.error(str(e))
