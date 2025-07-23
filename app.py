import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("🪙 Gold Price Forecasting App using ARIMA/SARIMA")

# ------------------ Load Gold Data ------------------
@st.cache_data
def load_gold_data():
    data = yf.download("GC=F", start="2005-01-01", interval="1mo")
    data = data[["Close"]].dropna()
    data.rename(columns={"Close": "Price"}, inplace=True)
    return data

gold = load_gold_data()

# ------------------ Sidebar ------------------
st.sidebar.header("📌 Forecast Settings")
model_type = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA"])
symbol = st.sidebar.text_input("Currency Symbol", "$")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

try:
    if model_type == "ARIMA":
        model = load_model("arima_gold_model.pkl")
    else:
        model = load_model("sarima_gold_model.pkl")
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Failed to load {model_type} model: {e}")
    model_loaded = False

# ------------------ Forecast Generation ------------------
if model_loaded:
    forecast_months = 36

    try:
        forecast = model.get_forecast(steps=forecast_months)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()

        last_date = gold.index[-1]
        forecast_index = pd.date_range(start=last_date + relativedelta(months=1), periods=forecast_months, freq='MS')

        forecast_df = pd.DataFrame({
            "Forecast": mean_forecast.values,
            "Lower CI": conf_int.iloc[:, 0].values,
            "Upper CI": conf_int.iloc[:, 1].values
        }, index=forecast_index)

        forecast_df["Month"] = forecast_df.index.strftime("%Y-%m")

    except Exception as e:
        st.error(f"⚠️ Error generating forecast: {e}")
        forecast_df = None

    # ------------------ Forecast Display ------------------
    if forecast_df is not None:
        start_date = forecast_df.index[0]
        end_date = forecast_df.index[-1]

        selected_date = st.date_input("📅 Select Forecast Month", min_value=start_date, max_value=end_date)
        selected_month = selected_date.strftime("%Y-%m")

        if selected_month in forecast_df["Month"].values:
            row = forecast_df[forecast_df["Month"] == selected_month].iloc[0]

            if pd.isna(row["Forecast"]):
                st.error("⚠️ Forecast data is NaN. Please retrain the model.")
            else:
                st.subheader(f"📅 Forecast for {selected_date.strftime('%B %Y')}")
                st.metric("Predicted Price", f"{symbol}{row['Forecast']:.2f}",
                          delta=f"± {(row['Upper CI'] - row['Lower CI']) / 2:.2f}")
        else:
            st.warning(
                f"⚠️ Selected date {selected_date.strftime('%B %Y')} is outside forecast range "
                f"({start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')})."
            )

        # ------------------ Forecast Table ------------------
        with st.expander("📊 View Full Forecast Table"):
            st.dataframe(forecast_df[["Forecast", "Lower CI", "Upper CI"]].style.format(precision=2))

        # ------------------ Forecast Plot ------------------
        st.subheader("📈 Forecast Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gold.index[-60:], gold["Price"].tail(60), label="Historical Price", color="blue")
        ax.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast", color="orange")
        ax.fill_between(forecast_df.index, forecast_df["Lower CI"], forecast_df["Upper CI"], alpha=0.2, color="orange")
        ax.set_title("Gold Price Forecast")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)
