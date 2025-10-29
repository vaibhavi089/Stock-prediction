import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import time

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Stock Price Prediction", page_icon=":chart_with_upwards_trend:", layout="centered")
st.markdown("### 📈 Predict Future Stock Prices using an LSTM Neural Network")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter stock symbol (e.g. AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))
train_button = st.sidebar.button("Train Model")

# ---------------- Data Loading ----------------
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end, retries=3):
    """
    Safely downloads stock data with retries and fallback handling.
    """
    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                threads=False,
                auto_adjust=False,
                prepost=True,
                repair=True
            )
            if data.empty:
                continue
            return data[['Close']]
        except Exception as e:
            time.sleep(2)
            if attempt == retries - 1:
                st.error(f"❌ Failed to fetch data for {ticker}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

data = load_data(ticker, start_date, end_date)

# ---------------- Handle empty data ----------------
if data.empty:
    st.warning("⚠️ No data available for this ticker or date range. Please try another combination.")
    st.stop()

# ---------------- Display data ----------------
st.subheader(f"{ticker} Stock Closing Prices")
st.line_chart(data['Close'])

# ---------------- Data Preparation ----------------
def prepare_data(data, seq_len=60):
    if len(data) < seq_len:
        st.error(f"❌ Not enough data to train (need at least {seq_len} days).")
        st.stop()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    x, y = [], []
    for i in range(seq_len, len(scaled)):
        x.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

# ---------------- LSTM Training ----------------
@st.cache_resource
def train_lstm_model(x_train, y_train, x_test, y_test):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(x_test)
    return model, predictions

# ---------------- Training + Prediction ----------------
if train_button:
    if len(data) < 100:
        st.warning("⚠️ Please select a longer date range (at least a few months).")
        st.stop()

    st.info("⏳ Training the LSTM model... please wait.")

    x, y, scaler = prepare_data(data.values)
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model, predictions = train_lstm_model(x_train, y_train, x_test, y_test)

    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual, label='Actual', linewidth=2)
    ax.plot(predictions, label='Predicted', linestyle='--')
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # ---------------- Next Day Prediction ----------------
    if len(data) >= 60:
        last_60 = data[-60:].values
        last_60_scaled = scaler.transform(last_60)
        x_input = np.reshape(last_60_scaled, (1, 60, 1))
        next_day_pred = model.predict(x_input)
        next_day_price = scaler.inverse_transform(next_day_pred)[0][0]
        st.success(f"📅 Predicted next day closing price: **${next_day_price:.2f}**")
    else:
        st.warning("⚠️ Not enough recent data to predict the next day price.")
