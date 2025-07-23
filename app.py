import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# ---------- Page Setup ----------
st.set_page_config(page_title="Gold Price Forecasting", layout="wide")
st.title("📈 Gold Price Forecasting App")
st.write("Predict gold prices using ARIMA or SARIMA models.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔧 Forecast Settings")
    model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA"])
    mode = st.radio("Forecast Mode", ["Months Ahead", "Specific Date"])
    currency = st.radio("Currency", ["USD ($)", "INR (₹)"])
    usd_to_inr = 83.0  # Static exchange rate

    forecast_months = 0
    selected_date = None
    max_forecast_date = (date.today() + relativedelta(months=36)).replace(day=1)

    if mode == "Months Ahead":
        forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    else:
        start_date = date.today()
        selected_date = st.date_input("Select Forecast End Date", min_value=start_date, max_value=max_forecast_date)
        if selected_date <= start_date:
            st.warning("Please select a future date.")
        else:
            delta = relativedelta(selected_date, start_date)
            forecast_months = delta.years * 12 + delta.months + (1 if delta.days > 0 else 0)

    if forecast_months > 0:
        st.info(f"Forecasting for {forecast_months} month(s)")

# ---------- Load Data ----------
@st.cache_data
def load_gold_data():
    data = yf.download("GC=F", start="2005-01-01", end=datetime.today(), interval="1mo")
    df = data[["Close"]].rename(columns={"Close": "Price"}).dropna()
    return df

# ---------- Plot Forecast ----------
def plot_forecast(history, forecast_df, model_name, currency):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.index, history['Price'], label="Historical Prices", linewidth=2)

    ax.plot(forecast_df.index, forecast_df['Forecast'], linestyle='--', label="Forecast", linewidth=2)
    ax.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], alpha=0.3, label="95% Confidence Interval")

    ax.set_title(f"{model_name} Forecast", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)" if "INR" in currency else "Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------- Load Model ----------
def load_model(model_name):
    filename = "arima_gold_model.pkl" if model_name == "ARIMA" else "sarima_gold_model.pkl"
    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading {model_name} model: {e}")
        return None

# ---------- Main Forecast Logic ----------
def main():
    if forecast_months <= 0:
        st.stop()

    gold = load_gold_data()

    # Convert to INR if selected
    if "INR" in currency:
        gold["Price"] *= usd_to_inr

    symbol = "₹" if "INR" in currency else "$"

    with st.expander("📜 View Historical Gold Prices"):
        st.dataframe(gold.style.format({"Price": f"{symbol}{{:,.2f}}"}))

    model = load_model(model_choice)
    if model is None:
        return

    try:
        forecast = model.get_forecast(steps=forecast_months)
        conf_int = forecast.conf_int()
        forecast_index = pd.date_range(start=gold.index[-1] + relativedelta(months=1), periods=forecast_months, freq='M')

        forecast_df = pd.DataFrame({
            "Forecast": forecast.predicted_mean,
            "Lower CI": conf_int.iloc[:, 0],
            "Upper CI": conf_int.iloc[:, 1]
        }, index=forecast_index)

        if "INR" in currency:
            forecast_df *= usd_to_inr

        forecast_df["Month"] = forecast_df.index.strftime("%Y-%m")

        # Show Forecast Table
        st.subheader(f"{model_choice} Forecast for Next {forecast_months} Month(s)")
        st.dataframe(
            forecast_df.drop(columns=["Month"]).style.format({
                "Forecast": f"{symbol}{{:,.2f}}",
                "Lower CI": f"{symbol}{{:,.2f}}",
                "Upper CI": f"{symbol}{{:,.2f}}"
            }), use_container_width=True
        )

        # Forecast Chart
        st.subheader("📈 Forecast Chart")
        plot_forecast(gold, forecast_df, model_choice, currency)

        # Model Summary
        with st.expander("📘 Model Summary"):
            try:
                st.text(model.summary())
            except:
                st.warning("Model summary not available.")

        # Specific Date Prediction
        if mode == "Specific Date" and selected_date:
            selected_month_key = selected_date.replace(day=1).strftime("%Y-%m")
            available_months = forecast_df["Month"].tolist()

            if selected_month_key in available_months:
                pred = forecast_df.loc[forecast_df["Month"] == selected_month_key].iloc[0]
                st.subheader(f"📅 Forecast for {selected_date.strftime('%B %Y')}")
                st.metric("Predicted Price", f"{symbol}{pred['Forecast']:.2f}",
                          delta=f"± {(pred['Upper CI'] - pred['Lower CI']) / 2:.2f}")
            else:
                st.warning(
                    f"Selected date **{selected_date.strftime('%B %Y')}** is outside the forecast range.\n\n"
                    f"Please choose a date between **{forecast_df.index[0].strftime('%B %Y')}** and **{forecast_df.index[-1].strftime('%B %Y')}**."
                )

    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")

# ---------- Run App ----------
if __name__ == "__main__":
    main()
