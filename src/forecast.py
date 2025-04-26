import numpy as np
import pandas as pd

def forecast_future(model, last_sequence, future_days, scaler):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :])[0,0]
        predictions.append(pred)

        new_entry = np.array([pred, current_sequence[-1, 1]])  # Keep Volume the same
        current_sequence = np.vstack((current_sequence[1:], new_entry))

    predictions = np.array(predictions).reshape(-1, 1)
    padded_predictions = np.hstack((predictions, np.zeros_like(predictions)))  # add dummy Volume
    future_prices = scaler.inverse_transform(padded_predictions)[:, 0]

    return future_prices
