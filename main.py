import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Streamlit setup
st.set_page_config(page_title="Stock Price Prediction", page_icon=":chart_with_upwards_trend:", layout="centered")
st.markdown("### Predict future stock prices using an LSTM Neural Network")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter stock symbol (e.g. AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))
train_button = st.sidebar.button("Train Model")


# -------- Data Loading -------- #
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.error("⚠️ No data found for this ticker. It may be delisted or invalid.")
            return pd.DataFrame({"Close": []})
        return data[["Close"]]
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame({"Close": []})



data = load_data(ticker, start_date, end_date)

if data.empty:
    st.warning("No data available for the selected ticker. Please try another one.")
    st.stop()

st.subheader(f"{ticker} Stock Closing Prices")
st.line_chart(data["Close"])


# -------- Data Preparation -------- #
def prepare_data(data, seq_len=60):
    """Prepare data for LSTM input."""
    if len(data) < seq_len + 1:
        st.error("❗ Not enough data to train the model. Try selecting a larger date range.")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    x, y = [], []
    for i in range(seq_len, len(scaled)):
        x.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler


# -------- Model Training -------- #
@st.cache_resource
def train_lstm_model(x_train, y_train, x_test):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(x_test)
    return model, predictions


# -------- Train Model -------- #
if train_button:
    st.write("⏳ Training or loading cached LSTM Model...")

    x, y, scaler = prepare_data(data.values)
    if x is None:
        st.stop()

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model, predictions = train_lstm_model(x_train, y_train, x_test)

    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Predict next day
    last_60 = data[-60:].values
    if len(last_60) == 60:
        last_60 = scaler.transform(last_60)
        x_input = np.reshape(last_60, (1, 60, 1))
        next_day_pred = model.predict(x_input)
        next_day_price = scaler.inverse_transform(next_day_pred)[0][0]
        st.success(f"📅 Predicted next day closing price: **${next_day_price:.2f}**")
    else:
        st.warning("Not enough recent data to predict the next day.")
