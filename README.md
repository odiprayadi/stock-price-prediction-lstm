# Stock Price Prediction using LSTM

This project uses deep learning (LSTM neural network) to predict stock prices based on historical Close price and Volume data from Yahoo Finance.

## Features
- Download historical stock data automatically using Yahoo Finance.
- Preprocessing and normalization of data (Close price and Volume).
- Build and train an LSTM model to predict future stock prices.
- Evaluate model performance using MAE, RMSE, and MAPE.
- Visualize actual vs predicted stock prices.
- Forecast future stock prices for a user-defined number of business days.

## Requirements
- Python 3.8+
- TensorFlow
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm
- pickle

You can install the required libraries via:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib tqdm tensorflow
