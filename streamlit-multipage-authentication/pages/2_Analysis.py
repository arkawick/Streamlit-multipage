import streamlit as st
import streamlit as st
import streamlit_authenticator as stauth
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
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

st.title("Project Analysis")
st.markdown("## Bar Graphs")

def plot_bar_chart(data, x, y, title):
    fig = px.bar(data, x=x, y=y, title=title)
    st.plotly_chart(fig)

################################################


# Example algorithms comparison 1
# algorithms1 = ['LSTM', 'CNN', 'Bidirectional LSTM']
# accuracy1 = [0.85, 0.80, 0.87]  # Dummy accuracy data
# algorithm_data1 = pd.DataFrame({
#     'Algorithm': algorithms1,
#     'Accuracy': accuracy1
# })
# st.header("Algorithms Used")
# plot_bar_chart(algorithm_data1, x='Algorithm', y='Accuracy', title="Algorithm Accuracy Comparison")

# Data for traditional ML methods
ml_algorithms = ['Linear regression', 'Logistic regression', 'K-Nearest Neighbor (KNN)', 'Gaussian Naïve Bayes']
ml_rmse = [8.9, 8.012, 29.069, 11.07]
ml_data = pd.DataFrame({
    'Algorithm': ml_algorithms,
    'RMSE': ml_rmse
})

st.header("RMSEs of Traditional Machine Learning Methods")
plot_bar_chart(ml_data, x='Algorithm', y='RMSE', title="Traditional ML Methods RMSE Comparison")

# Data for time series analysis methods
ts_algorithms = ['5-day SMA', '10-day SMA', '20-day SMA', 'ARIMA', 'Prophet']
ts_rmse = [6.8635, 16.6689, 26.54196, 11.12, 14.408]
ts_data = pd.DataFrame({
    'Algorithm': ts_algorithms,
    'RMSE': ts_rmse
})

st.header("RMSEs of Strictly Time Series Analysis Methods")
plot_bar_chart(ts_data, x='Algorithm', y='RMSE', title="Time Series Methods RMSE Comparison")

# Data for deep learning methods
dl_algorithms = ['Dense layer Forecast', 'Simple RNN', 'CNN', 'GRU', 'LSTM']
dl_rmse = [14.4141, 6.8712, 6.0812, 5.96762, 3.36544]
dl_data = pd.DataFrame({
    'Algorithm': dl_algorithms,
    'RMSE': dl_rmse
})

st.header("RMSEs of Deep Learning and Neural Networks")
plot_bar_chart(dl_data, x='Algorithm', y='RMSE', title="Deep Learning Methods RMSE Comparison")

# Data for LSTM variants
lstm_variants = [
    'LSTM(L-50,50,50|D-1)', 'LSTM(L-50,50,50|D-25,1)', 'LSTM(L-50,50|D-25,1)',
    'LSTM(L-128,64|D-25,1)', 'LSTM(L-256,128,64|D-25,1)', 'LSTM((L-100, d-0.2)*5|D-1)',
    'LSTM((L-100, d-0.2)*4|D-1)', 'LSTM((L-100, d-0.2)*3|D-1)'
]
lstm_rmse = [16.168, 11.169778, 2.9759, 3.3644, 3.59565, 4.50203, 6.25669, 5.35838]
lstm_data = pd.DataFrame({
    'Variant': lstm_variants,
    'RMSE': lstm_rmse
})

st.header("RMSEs of Variants of LSTM")
plot_bar_chart(lstm_data, x='Variant', y='RMSE', title="LSTM Variants RMSE Comparison")

# Data for BiLSTM variants
bilstm_variants = [
    'BiLSTM(B-256|D-25,1)', 'BiLSTM(B-128|D-25,1)', 'BiLSTM(B-128,64,d-0.5|D-25,1)', 'BiLSTM(B-128,64,32|D-25,1)'
]
bilstm_rmse = [10.3486, 8.12783, 3.286326, 3.276113]
bilstm_data = pd.DataFrame({
    'Variant': bilstm_variants,
    'RMSE': bilstm_rmse
})

st.header("RMSEs of Variants of BiLSTM")
plot_bar_chart(bilstm_data, x='Variant', y='RMSE', title="BiLSTM Variants RMSE Comparison")

# Data for Hybrid LSTMs
hybrid_variants = [
    'LSTM(L-50,50|D-25,1)', 'BiLSTM(B-128,64,32|D-25,1)', 'Hybrid LSTM(C-32|L-32,32|D-1)',
    'Hybrid LSTM (C-64,128,64|(L-100, d-0.2)*2|D-1)', 'Hybrid LSTM (C-64,128,64|(B-100, d-0.2)*2|D-1)'
]
hybrid_rmse = [2.9759, 3.276113, 3.5956, 2.28643, 2.2956]
hybrid_data = pd.DataFrame({
    'Variant': hybrid_variants,
    'RMSE': hybrid_rmse
})

st.header("RMSEs of Variants of LSTM, BiLSTM with respect to Hybrid LSTMs")
plot_bar_chart(hybrid_data, x='Variant', y='RMSE', title="Hybrid LSTMs RMSE Comparison")

st.header("Result Analysis")
st.markdown("""
After comparing Traditional Machine Learning Methods, Strictly Time Series Analysis Methods, 
and Deep Learning and Neural Networks, we have come to a conclusion that for this dataset, 
LSTM is the best model so far in these three broad categories of Algorithms. 

By further comparing the variants of LSTM, BiLSTM & Hybrid LSTMs in a trial-and-error manner, 
we tried to find the most accurate combination of layers which will give the best results 
and the least amount of error. The difference in the errors in the best models of LSTM, BiLSTM 
& Hybrid LSTM are really negligible, but the hybrid LSTM performed better than the other final models.

**Best Model:**
The results in various research papers show that additional training of data in BiLSTM-based modeling 
or the hybrid LSTM offers better predictions than regular LSTM-based models or the ARIMA models.
But in this case, the Hybrid CNN-LSTM model with dropout regularization performed the best, even better 
than the BiLSTM variant of this particular hybrid model, which is quite surprising.
""")
