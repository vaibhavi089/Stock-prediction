import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 LSTM Stock Price Predictor")


ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
train_button = st.sidebar.button("Train & Predict")


@st.cache_data
def load_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

if train_button:
    df = load_stock_data(ticker, start_date, end_date)
    st.subheader(f"📊 Historical Data for {ticker}")
    st.write(df.tail())

    
    df['Daily_Return'] = df['Close'].pct_change()
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['OC_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df.dropna(inplace=True)

    

    st.subheader("📉 Stock Closing Price & Moving Averages")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Close', color='blue')
    ax.plot(df.index, df['MA20'], label='MA20', color='orange', linestyle='--')
    ax.plot(df.index, df['MA50'], label='MA50', color='green', linestyle='--')
    ax.legend()
    ax.set_title(f"{ticker} Price Trend")
    st.pyplot(fig)

    
    class StockDataset(Dataset):
        def __init__(self, data, seq_len=60):
            self.seq_len = seq_len
            self.features = ['Close', 'Volume', 'Daily_Return', 'HL_Pct', 'OC_Pct', 'MA20', 'Volatility']
            self.data = data[self.features].values
            self.target = data['Close'].values.reshape(-1, 1)
            self.scaler = RobustScaler()
            self.data_scaled = self.scaler.fit_transform(self.data)
            self.target_scaled = self.scaler.fit_transform(self.target)

        def __len__(self):
            return len(self.data_scaled) - self.seq_len

        def __getitem__(self, idx):
            X = self.data_scaled[idx:idx+self.seq_len]
            y = self.target_scaled[idx+self.seq_len]
            return torch.FloatTensor(X), torch.FloatTensor(y)

    dataset = StockDataset(df)
    train_size = int(0.8 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    

    class StockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super(StockLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(input_size=7).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 15

    st.subheader("🚀 Training Progress")
    progress = st.progress(0)
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        progress.progress((epoch + 1) / epochs)

    st.success("✅ Model training completed!")

    

    fig2, ax2 = plt.subplots()
    ax2.plot(losses, color='purple')
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    st.pyplot(fig2)

    

    def predict_future_prices(model, dataset, days=30):
        model.eval()
        future_predictions = []

        last_idx = len(dataset) - 1
        last_sequence, _ = dataset[last_idx]
        current_sequence = last_sequence.clone().detach().to(device)

        with torch.no_grad():
            for _ in range(days):
                input_seq = current_sequence.unsqueeze(0)
                prediction = model(input_seq)
                future_predictions.append(prediction.item())

                new_row = current_sequence[-1].clone()
                new_row[0] = prediction.item()
                current_sequence = torch.cat([current_sequence[1:], new_row.unsqueeze(0)])

        return np.array(future_predictions)

    st.subheader("🔮 Predicting Future Stock Prices (Next 30 Days)")
    future_predictions = predict_future_prices(model, dataset, days=30)
    future_predictions_original = dataset.scaler.inverse_transform(
        future_predictions.reshape(-1, 1)).flatten()

    

    def create_static_future_plot(actual_prices, future_predictions):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(actual_prices)), actual_prices, 'b-', label='Historical Prices')
        ax.plot(range(len(actual_prices), len(actual_prices) + len(future_predictions)),
                future_predictions, 'r--', label='Future Predictions')
        ax.axvline(len(actual_prices) - 1, color='gray', linestyle=':', alpha=0.7)
        ax.legend()
        ax.set_title("Stock Price Forecast: Historical vs Future (Next 30 Days)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.grid(True)
        return fig

    actual_prices = df['Close'].values[-200:]  # 
    fig3 = create_static_future_plot(actual_prices, future_predictions_original)
    st.pyplot(fig3)

    ##forecast
    st.subheader("📅 Next 30 Days Forecast (in $)")
    forecast_df = pd.DataFrame({
        "Day": np.arange(1, 31),
        "Predicted Price": np.round(future_predictions_original, 2)
    })
    st.dataframe(forecast_df, use_container_width=True)
