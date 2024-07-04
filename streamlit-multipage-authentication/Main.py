import pickle
from pathlib import Path
from datetime import date, timedelta
import base64
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import yfinance as yf

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Streamlit Dashboard", page_icon=":bar_chart:", layout="wide")

hide_bar= """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        visibility:hidden;
        width: 0px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        visibility:hidden;
    }
    </style>
"""

# --- USER AUTHENTICATION ---
names = ["Arka", "Ayan", "Rohit"]
usernames = ["Arka", "Ayan", "Rohit"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "SIPL_dashboard", "abcdef")

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status == None:
    st.warning("Please enter your username and password")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status:
    # ---- SIDEBAR ----
    st.sidebar.title(f"Welcome {name}")
    st.write("# Welcome to Streamlit Stock Predictor & Visualizer")

    ###---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    authenticator.logout("Logout", "sidebar")

    # --- STOCK DATA SELECTION AND DISPLAY ---
    st.subheader("Stock Data Patterns")
    stock_option = st.text_input('Enter Stock Symbol:', 'AAPL').upper()

    # Fetch past month's data
    end_date = date.today()
    start_date_past_month = end_date - timedelta(days=30)
    data_past_month = yf.download(stock_option, start=start_date_past_month, end=end_date)

    st.subheader(f"{stock_option} Stock Data for the Past Month")
    st.line_chart(data_past_month['Close'])

    # Function to create download link for data
    def get_table_download_link(df, filename):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV file</a>'
        return href

    st.markdown(get_table_download_link(data_past_month, f"{stock_option}_past_month_stock_data"), unsafe_allow_html=True)

    # Fetch past year's data
    start_date_past_year = end_date - timedelta(days=365)
    data_past_year = yf.download(stock_option, start=start_date_past_year, end=end_date)

    st.subheader(f"{stock_option} Stock Data for the Past Year")
    st.line_chart(data_past_year['Close'])

    st.markdown(get_table_download_link(data_past_year, f"{stock_option}_past_year_stock_data"), unsafe_allow_html=True)

    # Fetch past 10 year data
    start_date_past_year = end_date - timedelta(days=3650)
    data_past_year = yf.download(stock_option, start=start_date_past_year, end=end_date)

    st.subheader(f"{stock_option} Stock Data for the Past 10 Years")
    st.line_chart(data_past_year['Close'])

    st.markdown(get_table_download_link(data_past_year, f"{stock_option}_past_year_stock_data"), unsafe_allow_html=True)
