# Demand Forecasting using Time Series Analysis

## Overview
Time series forecasting model to predict product demand using historical sales data. This project demonstrates how accurate forecasting enables optimal inventory management and supply chain planning.

## Objective
- Forecast future product demand accurately
- Identify seasonal patterns and trends
- Enable data-driven inventory decisions
- Optimize warehouse capacity and logistics

## Dataset
- **Source:** Daily sales data (E-commerce)
- **Size:** 730 daily observations (2 years)
- **Target:** Daily product demand
- **Frequency:** Daily time series

## Methodology

### 1. Time Series Decomposition
- Analyzed trend (long-term direction)
- Identified seasonality (weekly, monthly patterns)
- Examined residual noise

### 2. Stationarity Testing
- Augmented Dickey-Fuller (ADF) test
- Determined if differencing needed
- Validated assumption for ARIMA

### 3. ARIMA Modeling
- Auto-Regressive: Past values predict future
- Integrated: Differencing for stationarity
- Moving Average: Incorporating past errors
- Tuned (p,d,q) parameters

### 4. Prophet Forecasting
- Facebook's forecasting library
- Captures seasonality automatically
- Handles holidays and special events
- Provides confidence intervals

## Results

### Performance Metrics
| Metric | ARIMA | Prophet |
|--------|-------|---------|
| **RMSE** | 245.32 | 198.47 |
| **MAE** | 156.78 | 142.31 |
| **MAPE** | 9.2% | 8.1% |

### Key Findings
✅ **Seasonality:** Strong weekly pattern (weekends peak sales)
✅ **Trend:** Consistent upward trend over 2-year period
✅ **Accuracy:** Prophet RMSE of 198.47 units
✅ **MAPE:** 8.1% mean absolute percentage error (excellent)

### Business Impact
- **Inventory Optimization:** Predict demand → Stock optimal levels
- **Warehouse Planning:** Allocate capacity based on forecasts
- **Procurement:** Order from suppliers with accuracy
- **Staffing:** Schedule warehouse staff based on expected volume
- **Revenue Protection:** Prevent stockouts (lost sales) and overstock (waste)

## Files in This Project
demand-forecasting/
├── README.md # This file
├── sales_data.csv # Dataset (730 records)
├── timeseries_analysis.ipynb # Complete analysis notebook
├── data_preparation.py # Data loading and cleaning
├── arima_model.py # ARIMA forecasting
├── prophet_model.py # Prophet forecasting
└── requirements.txt # Python dependencies

## How to Run

### Option 1: View Analysis Online
Open `timeseries_analysis.ipynb` in GitHub to see complete analysis with visualizations.

### Option 2: Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare data:**
```bash
python data_preparation.py
```

3. **Run ARIMA model:**
```bash
python arima_model.py
```

4. **Run Prophet model:**
```bash
python prophet_model.py
```

5. **View notebook:**
```bash
jupyter notebook timeseries_analysis.ipynb
```

## Key Insights

### Demand Patterns
- **Weekly Seasonality:** Sales peak on weekends, dip on weekdays
- **Yearly Trend:** Consistent 15% YoY growth
- **Special Events:** Q4 spike (holiday season)
- **Volatility:** Lower volatility in mature months

### Model Comparison
- **ARIMA:** Better for stationary, linear patterns
- **Prophet:** Better for seasonal data with trend changes
- **Ensemble:** Could combine both for robustness

## Technologies Used

**Time Series Libraries:**
- statsmodels: ARIMA and ADF testing
- Prophet: Facebook's forecasting tool
- pandas: Time series manipulation

**Data & Analysis:**
- NumPy: Numerical computations
- scikit-learn: Data preprocessing

**Visualization:**
- Matplotlib: Time series plots
- Seaborn: Statistical plots

## Author
**Navya Ravindran**
- Business Analyst & Data Professional
- 8+ years analytics experience
- Specializing in time series and forecasting

## Contact
📧 navyar91@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/navya-ravindran-762a6072)

---

## Next Steps
- Deploy forecast model to production
- Monitor forecast accuracy monthly
- Retrain model with new data quarterly
- Implement automated alerts for demand spikes
- Integrate with inventory management system
- Track business impact on inventory costs

---

*Last Updated: August 3, 2026*
