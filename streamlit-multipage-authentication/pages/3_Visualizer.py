import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from datetime import date
import yfinance as yf
from prophet import Prophet
import base64
from io import BytesIO

st.title("Stock Visualizer")

# Sidebar configuration
def sidebar():
    st.sidebar.header("User Input")
    option = st.sidebar.selectbox('Select Stock', ('AAPL', 'MSFT', 'TSLA', 'META'))
    today = date.today()
    before = today - pd.DateOffset(days=700)
    start_date = st.sidebar.date_input('Start date', before)
    end_date = st.sidebar.date_input('End date', today)
    if start_date < end_date:
        st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    return option, start_date, end_date

# Download stock data
@st.cache(allow_output_mutation=True)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    data.reset_index(inplace=True)
    return data

# Function to plot Bollinger Bands
def plot_bollinger_bands(data):
    indicator_bb = BollingerBands(data['Close'])
    bb = data[['Close']].copy()
    bb['Bollinger_high'] = indicator_bb.bollinger_hband()
    bb['Bollinger_low'] = indicator_bb.bollinger_lband()
    st.write('Bollinger Bands of Stock Dataset')
    st.line_chart(bb)

# Function to plot MACD
def plot_macd(data):
    macd = MACD(data['Close']).macd()
    st.write('Stock Moving Average Convergence Divergence (MACD)')
    st.area_chart(macd)

# Function to plot RSI
def plot_rsi(data):
    rsi = RSIIndicator(data['Close']).rsi()
    st.write('Stock RSI ')
    st.line_chart(rsi)

# Function to create download link for data
def get_table_download_link(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="stock_data.xlsx">Download excel file</a>'

# Function to plot time series data
def plot_time_series(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to forecast stock prices
def forecast_stock(data, period):
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    return forecast, m

def main():
    option, start_date, end_date = sidebar()
    data = load_data(option, start_date, end_date)

    plot_bollinger_bands(data)
    plot_macd(data)
    plot_rsi(data)
    
    st.write('Recent Data Overview')
    st.dataframe(data.tail())
    st.markdown(get_table_download_link(data), unsafe_allow_html=True)

    start_date = "2015-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    full_data = load_data(option, start_date, end_date)

    plot_time_series(full_data)

    st.title("Stock Prediction")
    st.subheader("Choose Number of Years")
    n_years = st.slider("", 1, 5)
    period = n_years * 365

    forecast, model = forecast_stock(full_data, period)

    st.write('Forecasted data')
    st.write(forecast.tail())

    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)

if __name__ == "__main__":
    main()
