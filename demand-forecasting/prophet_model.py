import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DEMAND FORECASTING - PROPHET MODEL")
print("=" * 70)

# Load data
print("\n📥 Loading data...")
df = pd.read_csv('sales_data.csv')

# Prepare for Prophet
print("\n🔄 Preparing data for Prophet...")

# Rename columns to match Prophet format
if 'Date' in df.columns:
    df = df.rename(columns={'Date': 'ds'})
else:
    df['ds'] = pd.date_range(start='2024-01-01', periods=len(df))

if 'Sales' in df.columns:
    df = df.rename(columns={'Sales': 'y'})
elif 'Demand' in df.columns:
    df = df.rename(columns={'Demand': 'y'})
else:
    df = df.rename(columns={df.columns[-1]: 'y'})

# Keep only ds and y columns
df = df[['ds', 'y']]

print(f"✅ Data prepared: {len(df)} observations")
print(f"   Date range: {df['ds'].min()} to {df['ds'].max()}")

# Try to import Prophet
try:
    from prophet import Prophet
    
    print("\n🤖 Training Prophet model...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    print("✅ Model trained successfully!")
    
    # Forecast
    print("\n🔮 Generating 30-day forecast...")
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Extract test predictions
    test_forecast = forecast[-30:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    print(f"\n📊 Prophet Model Results:")
    print(f"   Forecast generated for next 30 days")
    print(f"   Average predicted demand: {test_forecast['yhat'].mean():.2f} units")
    print(f"   Forecast range: {test_forecast['yhat_lower'].min():.2f} to {test_forecast['yhat_upper'].max():.2f}")
    
    print(f"\n✅ Prophet forecasting complete!")
    print(f"   Model captures: Trend, Seasonality, Growth")
    
except ImportError:
    print("\n⚠️ Prophet not installed")
    print("   Install with: pip install prophet")
except Exception as e:
    print(f"❌ Error: {e}")
