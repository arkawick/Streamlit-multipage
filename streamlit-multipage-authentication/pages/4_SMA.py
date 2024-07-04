import os
import time
import requests
import pandas as pd
import numpy as np

import streamlit as st
import folium
from streamlit_folium import st_folium
import aspose.slides as slides


st.set_page_config(page_title="streamlit-folium documentation",page_icon="📈")


# st.header("Nvidia Jetson")
"# 📈 Moving Averages"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime

def display():
    # Initialize yfinance
    yf.pdr_override()

    # Title of the app
    st.title("Stock Price Prediction with Simple Moving Average")

    # Fetch data
    st.header("Fetching Data")
    symbol = st.text_input("Enter stock symbol (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start date", datetime(2012, 1, 1))
    end_date = st.date_input("End date", datetime.now())

    if st.button("Fetch Data"):
        df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        st.write(df.head())

        # Calculate SMAs
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()

        # Plot the closing price and SMAs using Plotly
        st.header("Simple Moving Averages")
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], mode='lines', name='SMA 5 Days'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], mode='lines', name='SMA 10 Days'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20 Days'))

        fig.update_layout(
            title=f'{symbol} Stock Price with SMAs',
            xaxis_title='Date',
            yaxis_title='Price USD ($)',
            legend_title='Legend',
            template='plotly_white'
        )

        st.plotly_chart(fig)

        # Predictions and RMSE (for simplicity, using SMA 5 as prediction example)
        actual_prices = df['Close'][20:]  # Skipping initial NaN values
        predicted_prices = df['SMA_5'][20:]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
        st.write(f"Root Mean Squared Error (SMA 5): {rmse}")

        st.header("Predicted vs Actual Prices")
        st.write(pd.DataFrame({
            'Actual Price': actual_prices,
            'Predicted Price (SMA 5)': predicted_prices
        }).tail(10))

        # Predict the closing price for the next day
        pred_price = df['SMA_5'].iloc[-1]
        st.write(f"Predicted Closing Price of {symbol} for next trading day (using SMA 5): {pred_price}")

        # Zoomable plot for recent data
        st.header("Recent Data Visualization")
        recent_data = df[-100:]  # Last 100 data points

        fig_recent = go.Figure()

        fig_recent.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], mode='lines', name='Closing Price'))
        fig_recent.add_trace(go.Scatter(x=recent_data.index, y=recent_data['SMA_5'], mode='lines', name='SMA 5 Days'))
        fig_recent.add_trace(go.Scatter(x=recent_data.index, y=recent_data['SMA_10'], mode='lines', name='SMA 10 Days'))
        fig_recent.add_trace(go.Scatter(x=recent_data.index, y=recent_data['SMA_20'], mode='lines', name='SMA 20 Days'))

        fig_recent.update_layout(
            title=f'Recent Data for {symbol} Stock Price with SMAs',
            xaxis_title='Date',
            yaxis_title='Price USD ($)',
            legend_title='Legend',
            template='plotly_white'
        )

        st.plotly_chart(fig_recent)

display()