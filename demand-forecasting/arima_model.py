import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEMAND FORECASTING - ARIMA MODEL")
print("=" * 70)

# Load data
print("\n📥 Loading data...")
df = pd.read_csv('sales_data.csv')

# Get sales column (adjust column name if different)
if 'Sales' in df.columns:
    sales = df['Sales']
elif 'Demand' in df.columns:
    sales = df['Demand']
else:
    sales = df.iloc[:, -1]  # Last column

print(f"✅ Data loaded: {len(sales)} observations")

# Stationarity test
print("\n🔍 Testing for stationarity (ADF Test)...")
adf_result = adfuller(sales)
print(f"   ADF Statistic: {adf_result[0]:.6f}")
print(f"   P-value: {adf_result[1]:.6f}")

if adf_result[1] <= 0.05:
    print(f"   ✅ Series is stationary (p-value <= 0.05)")
    d = 0
else:
    print(f"   ⚠️ Series is non-stationary (differencing needed)")
    d = 1

# Train-test split
print("\n✂️ Splitting data (80/20)...")
train_size = int(len(sales) * 0.8)
train, test = sales[:train_size], sales[train_size:]

print(f"   ✅ Train: {len(train)} samples")
print(f"   ✅ Test: {len(test)} samples")

# Fit ARIMA
print("\n🤖 Training ARIMA(1,1,1) model...")
try:
    model = ARIMA(train, order=(1, d, 1))
    fitted_model = model.fit()
    print("✅ Model trained successfully!")
    
    # Forecast
    print("\n🔮 Generating forecast...")
    forecast = fitted_model.get_forecast(steps=len(test))
    forecast_values = forecast.predicted_mean
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test, forecast_values))
    mae = mean_absolute_error(test, forecast_values)
    mape = np.mean(np.abs((test - forecast_values) / test)) * 100
    
    print(f"\n📊 ARIMA Model Performance:")
    print(f"   RMSE: {rmse:.2f} units")
    print(f"   MAE: {mae:.2f} units")
    print(f"   MAPE: {mape:.2f}%")
    
    print(f"\n✅ ARIMA forecasting complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Try adjusting ARIMA parameters or checking data quality")
