# Stock Predictor using LSTM

This project uses historical stock price data and a Long Short-Term Memory (LSTM) model to predict future stock prices. The model is trained on past stock data and can predict the next day's closing price based on the historical trends.

## Requirements

To run this project, you'll need the following Python libraries:

- `yfinance` - For downloading stock data.
- `numpy` - For numerical operations.
- `pandas` - For handling data.
- `sklearn` - For scaling the data.
- `tensorflow` - For building and training the LSTM model.
- `matplotlib` - For visualizing the predictions.

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Use
Download Historical Stock Data: The stock data is fetched using the yfinance library. You can change the stock symbol (e.g., AAPL for Apple) in the code.

Prepare Data: The data is normalized using MinMaxScaler and then transformed into sequences of past stock prices to predict the next day's closing price.

Train the Model: The model is built using an LSTM architecture with two LSTM layers and dropout for regularization. The model is trained using the training data.

Prediction and Visualization: After training, the model predicts stock prices on the test dataset, and the predictions are visualized alongside the actual stock prices.

## Customizing for Different Stocks
To predict stock prices for a different company, just modify the symbol variable in the Main Script section to the stock ticker you want to analyze (e.g., "TSLA" for Tesla).

python
Copy code
symbol = "TSLA"  # Example: Tesla stock
## Model Hyperparameters
You can adjust the following parameters for different results:

lookback: The number of past days the model will use to predict the next day's stock price.
LSTM model architecture: You can modify the number of LSTM layers, the number of units in each layer, and the dropout rate to improve model accuracy.
python
Copy code
lookback = 60  # Number of past days to use for prediction
## Example Output
The model generates a plot that compares the predicted stock prices with the actual stock prices:

Actual prices: Shown as the real historical closing prices.
Predicted prices: Shown as the model's predictions for the stock prices.
## Training and Evaluation
The model is trained with 80% of the historical data, and the remaining 20% is used for testing. Early stopping is used to prevent overfitting by halting training if the validation loss doesn't improve for a specified number of epochs.
