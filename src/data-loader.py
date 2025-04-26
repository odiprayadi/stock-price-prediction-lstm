import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start="2010-01-01"):
    data = yf.download(ticker, start=start)
    data = data[['Close', 'Volume']]
    data.dropna(inplace=True)
    return data
