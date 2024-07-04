import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
from contextlib import redirect_stdout
import io

# Initialize yfinance
yf.pdr_override()

# Set plot style
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Title of the app
st.title("Stock Price Prediction with LSTM")

# Fetch data
st.header("Fetching Data")
symbol = st.text_input("Enter stock symbol (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start date", datetime(2012, 1, 1))
end_date = st.date_input("End date", datetime.now())

if st.button("Fetch Data"):
    df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    st.write(df.head())

    # Plot the opening price
    st.header("Opening Price")
    plt.figure(figsize=(16, 8))
    plt.title('Opening Price')
    plt.plot(df['Open'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Open Price USD ($)', fontsize=18)
    st.pyplot(plt)

    # Data preprocessing
    data = df.filter(['Open'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Display the model layers
    st.header("Model Layers")
    model_layers = io.StringIO()
    with redirect_stdout(model_layers):
        model.summary()
    st.text(model_layers.getvalue())

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model and capture the terminal output
    st.header("Training the Model")
    train_output = io.StringIO()
    with redirect_stdout(train_output):
        history = model.fit(x_train, y_train, batch_size=1, epochs=3)
    st.text(train_output.getvalue())

    # Create the test data set
    test_data = scaled_data[training_data_len-60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate the RMSE
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    st.write(f"Root Mean Squared Error: {rmse}")

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    st.header("Model Predictions")
    plt.figure(figsize=(16, 8))
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Open Price USD ($)', fontsize=18)
    plt.plot(train['Open'], color='red')
    plt.plot(valid['Open'], color='yellow')
    plt.plot(valid['Predictions'], color='green')
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    st.pyplot(plt)

    st.write(valid.tail(15))

    # Predict the opening price for the next day
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    st.write(f"Predicted Opening Price of {symbol} for tomorrow: {pred_price[0][0]}")
