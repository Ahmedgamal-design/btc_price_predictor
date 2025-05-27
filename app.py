from flask import Flask, render_template_string
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# HTML Template
TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>BTC Price Predictor</title>
</head>
<body>
    <h1>Bitcoin Price Prediction</h1>
    <p>Predicted price after 3 hours: <strong>${{ prediction:.2f }}</strong></p>
    <img src="data:image/png;base64,{{ plot_url }}"/>
</body>
</html>
"""

# Fetch BTC data from CoinGecko API
def fetch_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "hourly"}
    response = requests.get(url, params=params)
    data = response.json()

    if "prices" not in data:
        raise KeyError("Missing 'prices' in response")

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# Train model and predict next price after 3 hours
def predict_price(df):
    df = df.copy()
    df["time"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["time"]], df["price"])
    future_time = [[len(df) + 3]]  # Prediction after 3 time units (3 hours)
    prediction = model.predict(future_time)
    return prediction[0]

# Plot BTC price chart
def plot_prices(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    df["price"].plot(ax=ax, title="BTC Price (Last 24 hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

@app.route("/")
def index():
    try:
        df = fetch_btc_data()
        prediction = predict_price(df)
        plot_url = plot_prices(df)
        return render_template_string(TEMPLATE, prediction=prediction, plot_url=plot_url)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)