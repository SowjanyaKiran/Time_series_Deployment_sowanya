import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecasting App")
st.write("Predict future gold prices using ARIMA or SARIMA models.")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("🔧 Forecast Settings")
    model_map = {
        "ARIMA": "arima_gold_model.pkl",
        "SARIMA": "sarima_gold_model.pkl"
    }
    selected_model_name = st.selectbox("Choose Model", list(model_map.keys()))
    model_file = model_map[selected_model_name]

    mode = st.radio("Forecast by:", ["Months Ahead", "Specific Date"])
    selected_date = None

    if mode == "Months Ahead":
        selected_months = st.slider("Forecast Period (months)", 1, 36, 12)
    else:
        selected_date = st.date_input("Select a future date", min_value=datetime.today())

    st.divider()
    currency = st.radio("Currency", ["USD ($)", "INR (₹)"])
    usd_to_inr = 83.0  # Approximate exchange rate

# ------------------ Data Loader ------------------
@st.cache_data
def load_gold_data():
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download("GC=F", start="2005-01-01", end=end_date, interval="1mo")
    return data["Close"].dropna()

# ------------------ Forecast Plotter ------------------
def plot_forecast(history, forecast_df, model_name, currency):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.index, history, label="Historical Prices", linewidth=2)
    ax.plot(forecast_df.index, forecast_df["Forecast"], linestyle='--', label="Forecast", linewidth=2)
    ax.fill_between(forecast_df.index, forecast_df["Lower CI"], forecast_df["Upper CI"], alpha=0.3, label="95% Confidence Interval")
    ax.set_title(f"{model_name} Forecast", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)" if "INR" in currency else "Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ------------------ Main App Logic ------------------
def main():
    gold = load_gold_data()
    last_date = gold.index[-1]
    forecast_months = 36  # Always forecast 36 months

    # Show historical data
    with st.expander("📜 View Historical Gold Prices"):
        df = gold.copy()

        if "INR" in currency:
            df *= usd_to_inr

        symbol = "₹" if "INR" in currency else "$"

        if isinstance(df, pd.Series):
            df_display = df.to_frame(name="Price")
        else:
            df_display = df.copy()
            if df_display.columns[0] != "Price":
                df_display.columns = ["Price"]

        st.dataframe(
            df_display.style.format({"Price": lambda x: f"{symbol}{x:,.2f}"}),
            use_container_width=True
        )

    # Load model
    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return

    # Forecast
    try:
        forecast = model.get_forecast(steps=forecast_months)
        conf_int = forecast.conf_int()
        future_index = pd.date_range(start=last_date + relativedelta(months=1), periods=forecast_months, freq='M')

        forecast_df = pd.DataFrame({
            "Forecast": forecast.predicted_mean,
            "Lower CI": conf_int.iloc[:, 0],
            "Upper CI": conf_int.iloc[:, 1]
        }, index=future_index)

        if "INR" in currency:
            forecast_df *= usd_to_inr

        forecast_df["Month"] = forecast_df.index.strftime("%Y-%m")
        symbol = "₹" if "INR" in currency else "$"

        # ---------- Specific Date Mode ----------
        if mode == "Specific Date" and selected_date:
            selected_key = selected_date.replace(day=1).strftime("%Y-%m")
            forecast_months_available = forecast_df["Month"].tolist()

            if selected_key in forecast_months_available:
            # selected_key = selected_date.strftime("%Y-%m")
            # if selected_key in forecast_df["Month"].values:
            #     pred = forecast_df[forecast_df["Month"] == selected_key].iloc[0]
                st.subheader(f"📅 Predicted Gold Price for {selected_date.strftime('%B %Y')}")
                st.metric("Forecast", f"{symbol}{pred['Forecast']:.2f}",
                          delta=f"± {(pred['Upper CI'] - pred['Lower CI']) / 2:.2f}")
            else:
                st.warning("⚠️ Forecast for selected month is not available (limit: next 36 months).")

        # ---------- Months Ahead Mode ----------
        elif mode == "Months Ahead":
            limited_df = forecast_df.head(selected_months)
            st.subheader(f"📊 Forecast for {selected_months} month(s)")
            st.dataframe(
                limited_df.drop(columns=["Month"]).style.format({
                    "Forecast": lambda x: f"{symbol}{x:,.2f}",
                    "Lower CI": lambda x: f"{symbol}{x:,.2f}",
                    "Upper CI": lambda x: f"{symbol}{x:,.2f}",
                }), use_container_width=True
            )

        # Plot
        st.subheader("📈 Forecast Chart")
        plot_forecast(gold, forecast_df, selected_model_name, currency)

        # Model Summary
        with st.expander("📘 Model Summary"):
            try:
                st.text(model.summary())
            except:
                st.info("Model summary not available.")

    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")

# ------------------ Entry ------------------
if __name__ == "__main__":
    main()
