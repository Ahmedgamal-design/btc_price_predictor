import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="BTC Price Prediction", layout="centered")
st.title("Bitcoin Price Prediction")

API_KEY = "adb2e003-6773-4184-94f4-6578bae25354"

@st.cache_data
def fetch_btc_history():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    headers = {"X-CMC_PRO_API_KEY": API_KEY}
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "time_period": "hourly",
        "interval": "hourly",
        "count": 24
    }
    r = requests.get(url, headers=headers, params=params)
    data = r.json()

    if "data" not in data or "quotes" not in data["data"]:
        raise ValueError("Failed to fetch data.")

    quotes = data["data"]["quotes"]
    df = pd.DataFrame(quotes)
    df["price"] = df["quote"].apply(lambda x: x["USD"]["close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df[["price"]]

def predict_next_price(df):
    df = df.copy()
    df["t"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["t"]], df["price"])
    next_t = [[len(df) + 3]]
    pred = model.predict(next_t)
    return pred[0]

try:
    df = fetch_btc_history()
    prediction = predict_next_price(df)

    st.subheader("Predicted price after 3 hours:")
    st.success(f"${prediction:,.2f}")

    st.subheader("Price chart:")
    fig, ax = plt.subplots()
    df["price"].plot(ax=ax, title="BTC Price (Last 24 hours)")
    ax.set_ylabel("USD")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
