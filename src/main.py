import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_stock_data
from model import prepare_data, build_lstm_model
from forecast import forecast_future

def main():
    ticker = input("Enter stock ticker: ").upper()
    future_days = int(input("Enter number of future business days to predict: "))

    data = load_stock_data(ticker)
    print(f"Data shape: {data.shape}")

    look_back = 60
    X, y, scaler = prepare_data(data, look_back)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(X_train.shape[1:])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros_like(predictions))))[:, 0]
    y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros_like(y_test.reshape(-1,1)))))[:, 0]

    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    mape = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Plot
    plt.figure(figsize=(14,7))
    plt.plot(data.index[-len(y_test):], y_test_rescaled, label="Actual")
    plt.plot(data.index[-len(y_test):], predictions_rescaled, label="Predicted")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Forecast
    last_sequence = X[-1]
    future_prices = forecast_future(model, last_sequence, future_days, scaler)

    future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'Date': future_dates, 'Forecasted Close': future_prices})
    print(future_df)

    # Plot future
    plt.figure(figsize=(14,7))
    plt.plot(data.index, data['Close'], label="Historical Close")
    plt.plot(future_df['Date'], future_df['Forecasted Close'], label="Forecasted Future", linestyle='--')
    plt.title(f"{ticker} Future Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
