# Model Description

## Project Overview
This project is a stock price trend prediction system.  
It uses historical stock price data to predict whether the stock price will go **up** or **down** the next day.  
The model is trained on features derived from Open, High, Low, Close, and Volume data.

---

## Data
- Source: [Yahoo Finance](https://finance.yahoo.com/)
- Features Used: Open, High, Low, Close, Volume
- Labels:
  - 1 = Price will go **up** tomorrow
  - 0 = Price will go **down** tomorrow
- Time window: Past 5 days' data used to predict the next day's movement.

---

## Preprocessing
- Fetch historical stock data.
- Calculate labels (next day's Close vs today's Close).
- Scale features using `StandardScaler`.
- Reshape data for CNN input:
  - Shape: (samples, 5, 5, 1)
  - (5 days, 5 features, 1 channel)

---

## Model Architecture
- Base: **Convolutional Neural Network (CNN)**
- Layers:
  - Conv2D (32 filters, 2x2 kernel, ReLU)
  - MaxPooling2D (2x2)
  - Flatten
  - Dense (64 units, ReLU)
  - Output Dense (1 unit, Sigmoid)

---

## Training
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Epochs: 50
- Batch size: 32
- Validation split: 20%

---

## Evaluation
- Accuracy is used as the primary evaluation metric.
- Predictions are tested on unseen data (test set).
- Additionally, a simple strategy is simulated to see if the model's predictions could lead to profit.

---

## How to Use
- Fetch new stock data.
- Preprocess and format it like the training data.
- Load the trained model.
- Predict whether the price will go up or down.
- Optionally: simulate trading strategy based on predictions.

---

## Requirements
See `requirements.txt` for the full list of Python libraries needed.

---
