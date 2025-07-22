import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Page config
st.set_page_config(page_title="Gold Price Forecasting", layout="wide")

st.title("Gold Price Forecasting App")
st.write("Forecasting gold prices using ARIMA and SARIMA models.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    # Forecast mode selection
    option = st.radio("Select Forecast Mode", ["By Months", "By End Date"])

    if option == "By Months":
        forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    else:
        start_date = datetime.today().date()
        forecast_end = st.date_input("Select Forecast End Date", min_value=start_date)

        if forecast_end <= start_date:
            st.warning("Please select a future date for forecasting.")
            forecast_months = 0
        else:
            delta = relativedelta(forecast_end, start_date)
            forecast_months = delta.years * 12 + delta.months + (1 if delta.days > 0 else 0)

    if forecast_months > 0:
        st.info(f"Forecasting for {forecast_months} month(s)")
    else:
        st.error("Invalid forecast period. Please adjust your input.")

# Load gold price data
@st.cache_data
def load_gold_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download("GC=F", start="2005-01-01", end=end_date, interval="1mo")
    df = data[['Close']].copy()
    df.columns = ['Price']
    df.dropna(inplace=True)
    return df

# Forecast plot
def plot_forecast(gold, forecast, conf_int, forecast_months, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gold.index, gold['Price'], label='Historical Prices', linewidth=2)

    forecast_index = pd.date_range(
        start=gold.index[-1] + relativedelta(months=1),
        periods=forecast_months,
        freq='M'
    )
    ax.plot(forecast_index, forecast.predicted_mean, label=f'{model_name} Forecast', linestyle='--')
    ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0
