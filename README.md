#  Stock Price Prediction (LSTM + Streamlit)

This app predicts the next-day closing price of any stock using an LSTM neural network.

###  Features
- Fetches real stock data via Yahoo Finance (`yfinance`)
- Trains an LSTM deep learning model
- Visualizes actual vs predicted prices
- Predicts next-day closing price
- Built with Streamlit

###  Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
