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
    params = {
        "vs_currency": "usd",
        "days": "1",
        "interval": "hourly"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = data.get("prices")

        if not prices:
            raise ValueError("No 'prices' data found in API response.")

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP Error: {err}")
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

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
    df["price"].plot(ax=ax, title="BTC Price (Last 24 Hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    st.pyplot(fig)

df = fetch_btc_data()

if df is not None:
    prediction = predict_price(df)
    st.subheader(f"Predicted Price After 3 Hours: ${prediction:,.2f}")
    plot_prices(df)
