import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Page configuration
st.set_page_config(page_title="Gold Price Forecasting", layout="wide")

st.title("Gold Price Forecasting App")
st.write("Forecasting gold prices using ARIMA and SARIMA models.")

# Sidebar for user input
with st.sidebar:
    st.header("Settings")

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

# Load historical gold price data
@st.cache_data
def load_gold_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download("GC=F", start="2005-01-01", end=end_date, interval="1mo")
    df = data[['Close']].copy()
    df.columns = ['Price']
    df.dropna(inplace=True)
    return df

# Plotting function
def plot_forecast(gold, forecast, conf_int, forecast_months, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gold.index, gold['Price'], label='Historical Prices', linewidth=2)

    forecast_index = pd.date_range(
        start=gold.index[-1] + relativedelta(months=1),
        periods=forecast_months,
        freq='M'
    )
    ax.plot(forecast_index, forecast.predicted_mean, label=f'{model_name} Forecast', linestyle='--')
    ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.2)

    ax.set_title(f"{model_name} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to handle model output
def display_model_results(model, gold, model_name):
    forecast = model.get_forecast(steps=forecast_months)
    conf_int = forecast.conf_int()

    forecast_index = pd.date_range(
        start=gold.index[-1] + relativedelta(months=1),
        periods=forecast_months,
        freq='M'
    )
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecast': forecast.predicted_mean,
        'Lower CI': conf_int.iloc[:, 0],
        'Upper CI': conf_int.iloc[:, 1]
    }).set_index('Date')

    st.subheader(f"{model_name} - {forecast_months}-Month Forecast")
    st.dataframe(forecast_df.style.format({
        "Forecast": "${:,.2f}",
        "Lower CI": "${:,.2f}",
        "Upper CI": "${:,.2f}"
    }))

    fig = plot_forecast(gold, forecast, conf_int, forecast_months, model_name)
    st.pyplot(fig)

    with st.expander(f"{model_name} Model Summary"):
        try:
            st.text(model.summary())
        except:
            st.warning("Model summary not available.")

# Main execution
def main():
    if forecast_months <= 0:
        st.stop()  # Stop the app if forecast period is invalid

    gold = load_gold_data()

    with st.expander("View Historical Gold Prices"):
        st.dataframe(gold.style.format({"Price": "${:,.2f}"}))

    # Load ARIMA model
    try:
        with open("arima_gold_model.pkl", "rb") as f:
            arima_model = pickle.load(f)
        st.success("ARIMA model loaded successfully")
        display_model_results(arima_model, gold, "ARIMA")
    except Exception as e:
        st.error(f"Error loading ARIMA model: {str(e)}")

    # Load SARIMA model
    try:
        with open("sarima_gold_model.pkl", "rb") as f:
            sarima_model = pickle.load(f)
        st.success("SARIMA model loaded successfully")
        display_model_results(sarima_model, gold, "SARIMA")
    except Exception as e:
        st.error(f"Error loading SARIMA model: {str(e)}")

if __name__ == "__main__":
    main()
