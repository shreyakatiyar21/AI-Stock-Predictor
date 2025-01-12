
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Step 1: Fetch Historical Stock Data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start = start_date, end = end_date)
    return stock_data[['Close']]


# Step 2: Prepare the Data
def prepare_data(data, lookback = 60):
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Step 3: Build the LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences = True, input_shape = input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences = False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model


# Step 4: Train the Model
def train_model(model, X_train, y_train, batch_size = 32, epochs = 50, validation_split = 0.2):
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
    history = model.fit(
        X_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split = validation_split,
        callbacks = [early_stopping]
    )
    return history


# Step 5: Predict and Visualize Results
def predict_and_plot(model, data, X_test, y_test, scaler, lookback = 60):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize predictions vs actual
    plt.figure(figsize = (10, 6))
    plt.plot(data.index[-len(y_test):], y_test, label = "Actual")
    plt.plot(data.index[-len(y_test):], predictions, label = "Predicted")
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()


# Main Script
if __name__ == "__main__":
    symbol = "AAPL"  # Example: Apple stock
    start_date = "2015-01-01"
    end_date = "2023-12-31"
    lookback = 60  # Number of days to look back

    # Fetch data
    data = fetch_stock_data(symbol, start_date, end_date)

    # Prepare data
    X, y, scaler = prepare_data(data.values, lookback)
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build and train model
    model = build_lstm_model((X_train.shape[1], 1))
    train_model(model, X_train, y_train)

    # Predict and visualize
    predict_and_plot(model, data, X_test, y_test, scaler, lookback)
