 #Gold Price Forecasting Using Time Series Analysis (ARIMA & SARIMA)#
This project delves into time series analysis and forecasting of monthly gold prices utilizing the ARIMA and SARIMA models. It comes equipped with a Streamlit web application that enables interactive exploration, model selection, and dynamic forecasting.

Project Highlights
Time Series Workflow:

Data Acquisition: Fetch gold price data via Yahoo Finance

Trend Visualization: Plot time-based price movements

STL Decomposition: Separate trend, seasonality, and noise

Stationarity Testing: Apply ADF test to check stationarity

Differencing: Stationarize the series

ACF & PACF Plots: Identify model orders

Model Building: Implement ARIMA & SARIMA models

Forecasting: Predict the next 12 months

Model Comparison: Evaluate models using AIC/BIC

Model Persistence: Save models with pickle

Streamlit Web Application:

Model Selection: Choose between ARIMA and SARIMA

Forecasting: Predict for up to 36 months ahead

Visualization: Display forecasts with confidence intervals

Raw Data: Access historical data and model summary

Repository Structure
bash
Copy
📁 gold-price-forecasting/
├── app.py                   # Main Streamlit app file
├── arima_gold_model.pkl      # Trained ARIMA model
├── sarima_gold_model.pkl     # Trained SARIMA model
├── model_training.ipynb      # Notebook for analysis and model training
├── requirements.txt          # List of dependencies
└── README.md                 # Project description and usage
How to Get Started
Clone the Repository

bash
Copy
git clone https://github.com/your-username/gold-price-forecasting.git
cd gold-price-forecasting
Install Dependencies
Use pip to install required libraries:

nginx
Copy
pip install -r requirements.txt
Run the Streamlit App
Launch the app using:

arduino
Copy
streamlit run app.py
Example Forecast Visualization
Check out an example forecast using the SARIMA model, along with a 95% confidence interval:
Example Forecast

Model Details
Model	AIC/BIC	Seasonality	Stationarity	Forecast Horizon
ARIMA	Evaluated	❌ No	Differenced	Short-term
SARIMA	Evaluated	✅ Yes	Differenced	Seasonal-aware

ACF & PACF Interpretation
ACF (Autocorrelation Function):

Spikes at seasonal lags indicate the need for seasonal terms in SARIMA.

Cut-off after lag p suggests the AR order.

Cut-off after lag q suggests the MA order.

Sample ADF Test Output
yaml
Copy
ADF Statistic: -1.56
p-value: 0.51
Conclusion: Time series is **non-stationary**. Differencing required.
Technologies Used
Python: Core programming language

Pandas & Matplotlib: Data manipulation and visualization

Statsmodels: ARIMA and SARIMA model implementation

yfinance: Fetching financial data

Streamlit: User interface for model interaction

Pickle: Model serialization for persistence

License
This project is open-source, licensed under the MIT License.

Author
Developed by [Sowjanya Kiran]
📧 Email: usowjanyakiran@gmail.com
🌐 GitHub: SowjanyaKiran/Time_series_Deployment_sowanya

Related Projects
Time Series Forecasting with Prophet

Stock Price Prediction using LSTM

requirements.txt
If you're looking to deploy or share this project, use this list in your requirements.txt:

streamlit
pandas
matplotlib
statsmodels
yfinance
python-dateutil
