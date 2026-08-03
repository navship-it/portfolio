import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("DEMAND FORECASTING - DATA PREPARATION")
print("=" * 60)

# Load data
print("\n📥 Loading dataset...")
df = pd.read_csv('sales_data.csv')

print(f"✅ Dataset loaded!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📊 First few rows:")
print(df.head())

# Check for missing values
print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")

# Convert to datetime if needed
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    print(f"✅ Date column converted")

# Basic statistics
print(f"\n📈 Sales Statistics:")
if 'Sales' in df.columns:
    print(f"   Mean: {df['Sales'].mean():.2f}")
    print(f"   Std Dev: {df['Sales'].std():.2f}")
    print(f"   Min: {df['Sales'].min():.2f}")
    print(f"   Max: {df['Sales'].max():.2f}")

print(f"\n✅ Data preparation complete!")
print(f"   Ready for time series analysis")
