import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Basic page config
st.set_page_config(page_title="Gold Price Forecasting", layout="wide")

# Title and description
st.title("Gold Price Forecasting App")
st.write("This app forecasts gold prices using time series models (ARIMA/SARIMA).")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Select Model", ["arima_gold_model.pkl", "sarima_gold_model.pkl"])
    forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    st.write("ARIMA: Short-term forecasts")
    st.write("SARIMA: Seasonal patterns")

# Load data function
@st.cache_data
def load_gold_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    gold_data = yf.download("GC=F", start="2005-01-01", end=end_date, interval="1mo")
    gold = gold_data[['Close']].copy()
    gold.columns = ['Price']
    gold.dropna(inplace=True)
    return gold

def plot_forecast(gold, forecast, conf_int, forecast_months, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(gold.index, gold['Price'], label='Historical Prices', linewidth=2)
    
    # Plot forecast
    forecast_index = pd.date_range(
        start=gold.index[-1] + relativedelta(months=1),
        periods=forecast_months,
        freq='M'
    )
    ax.plot(forecast_index, forecast.predicted_mean, label='Forecast', linestyle='--', linewidth=2)
    
    # Plot confidence interval
    ax.fill_between(forecast_index,
                   conf_int.iloc[:, 0],
                   conf_int.iloc[:, 1],
                   alpha=0.3,
                   label='95% Confidence Interval')
    
    # Formatting
    ax.set_title(f"Gold Price Forecast - {model_name.upper()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def main():
    try:
        # Load historical data
        gold = load_gold_data()
        
        # Display raw data
        with st.expander("View Historical Gold Prices"):
            st.dataframe(gold.style.format({"Price": "${:,.2f}"}))
        
        # Load model and make forecasts
        try:
            with open(model_choice, 'rb') as f:
                model = pickle.load(f)
            
            model_name = model_choice.split('_')[0].upper()
            st.success(f"Loaded {model_name} model")
            
            # Generate forecast
            forecast = model.get_forecast(steps=forecast_months)
            conf_int = forecast.conf_int()
            
            # Create forecast dataframe
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
            
            # Display forecast
            st.subheader(f"{forecast_months}-Month Forecast")
            st.dataframe(forecast_df.style.format({
                "Forecast": "${:,.2f}", 
                "Lower CI": "${:,.2f}", 
                "Upper CI": "${:,.2f}"
            }))
            
            # Plot forecast
            st.subheader("Forecast Visualization")
            fig = plot_forecast(gold, forecast, conf_int, forecast_months, model_name)
            st.pyplot(fig)
            
            # Show model summary
            with st.expander("Model Summary"):
                try:
                    st.text(model.summary())
                except:
                    st.warning("Model summary not available")
        
        except FileNotFoundError:
            st.error(f"Model file '{model_choice}' not found")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
