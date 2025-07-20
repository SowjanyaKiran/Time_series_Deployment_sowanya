# 🪙 Gold Price Forecasting Using Time Series Analysis (ARIMA & SARIMA)

This project performs in-depth **time series analysis** and **forecasting** on monthly gold prices using **ARIMA** and **SARIMA** models. It also features a full-fledged **Streamlit app** for interactive exploration, model selection, and dynamic forecasting.

---

## 📦 Project Features

### ✅ Time Series Workflow Includes:
- Downloading gold price data from **Yahoo Finance**
- **Visualization** of trends over time
- **STL decomposition** to separate trend, seasonality, and noise
- **ADF test** for stationarity
- **Differencing** to make the series stationary
- **ACF and PACF plots** to determine model orders
- **Model building** with ARIMA and SARIMA
- **Forecasting** next 12 months
- **Model comparison** using AIC/BIC
- **Model saving** using pickle

### ✅ Interactive Streamlit Web App:
- Choose between ARIMA and SARIMA models
- Forecast up to 36 months into the future
- Visualize forecast with **confidence intervals**
- View model summary and raw historical data

---

## 🗃️ Repository Structure

```

📁 gold-price-forecasting/
├── app.py                   # Streamlit app
├── arima\_gold\_model.pkl     # Trained ARIMA model (auto-generated)
├── sarima\_gold\_model.pkl    # Trained SARIMA model (auto-generated)
├── model\_training.ipynb     # Main analysis & modeling notebook (optional)
├── requirements.txt         # Required packages
└── README.md                # Project overview

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gold-price-forecasting.git
cd gold-price-forecasting
````

### 2. Install Dependencies

Install required libraries using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Example Forecast Visualization

*Forecast using SARIMA model (green) and 95% confidence interval:*
(https://github.com/SowjanyaKiran/Arima_Sarima_Time_series_Deployment_sowanya/)

---

## 🧠 Model Details

| Model  | AIC / BIC | Seasonality | Stationarity | Forecast Horizon |
| ------ | --------- | ----------- | ------------ | ---------------- |
| ARIMA  | Evaluated | ❌ No        | Differenced  | Short-term       |
| SARIMA | Evaluated | ✅ Yes       | Differenced  | Seasonal-aware   |

### 🔬 ACF & PACF Interpretation Tips:

* ACF with spikes at seasonal lags → Add seasonal terms
* PACF cut-off after lag `p` → Suggests AR model order
* ACF cut-off after lag `q` → Suggests MA model order

---

## 🧪 ADF Test Result Sample Output

```
ADF Statistic: -1.56
p-value: 0.51
Conclusion: Time series is **non-stationary**. Differencing is required.
```

---

## 🧰 Technologies Used

* `Python`
* `Pandas`, `Matplotlib`, `Statsmodels`
* `yfinance` for data fetching
* `Streamlit` for UI
* `Pickle` for model persistence

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

Developed by **\[Sowjanya Kiran]**
📧 usowjanyakiran@gmail.com
🌐 https://github.com/SowjanyaKiran/Time_series_Deployment_sowanya/

---

## 🔗 Related Projects

* Time Series Forecasting with Prophet
* Stock Price Prediction using LSTM

```

---

## 📌 `requirements.txt`

If you want to deploy or share the project, use this for your `requirements.txt`:

```

streamlit
pandas
matplotlib
statsmodels
yfinance
python-dateutil

```


