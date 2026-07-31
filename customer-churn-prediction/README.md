# Customer Churn Prediction

## Overview
Machine learning model to predict which subscription customers will cancel (churn). This project demonstrates end-to-end machine learning workflow: from data exploration to model training and evaluation.

## Objective
- Identify customers at risk of churning
- Understand key drivers of customer churn
- Build predictive model for proactive retention strategies
- Enable data-driven business decisions

## Dataset
- **Source:** Telecom Customer Churn (Kaggle)
- **Size:** 7,043 customers, 20 features
- **Target:** Binary classification (Churn: Yes/No)
- **Churn Rate:** ~26.5% (1,869 churned customers)

## Methodology

### 1. Exploratory Data Analysis
- Analyzed customer demographics and service usage patterns
- Identified churn distribution and key risk factors
- Visualized correlations and relationships in data

### 2. Data Preprocessing
- Handled missing values and outliers
- Encoded categorical features
- Scaled numerical features using StandardScaler
- Split data: 80% train, 20% test (stratified)

### 3. Model Training
Trained and compared two classification models:

**Model 1: Logistic Regression**
- Fast, interpretable baseline model
- ROC-AUC: **0.8392**

**Model 2: Random Forest**
- Ensemble method capturing complex patterns
- ROC-AUC: **0.8473** ⭐ (Best Model)

## Results

### Performance Metrics
| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Accuracy** | 0.7944 | 0.8011 |
| **Precision** | 0.6557 | 0.6778 |
| **Recall** | 0.5206 | 0.5731 |
| **F1-Score** | 0.5812 | 0.6214 |
| **ROC-AUC** | 0.8392 | **0.8473** |

### Key Findings
✅ **Model Accuracy:** Random Forest achieves 80.11% accuracy on test data
✅ **Churn Detection:** Identifies ~57% of customers who will churn (recall = 0.5731)
✅ **Precision:** 67.78% of predicted churners actually churn (precision = 0.6778)
✅ **ROC-AUC:** 0.8473 indicates excellent discrimination between classes

### Business Impact
- **Proactive Retention:** Identify high-risk customers before they churn
- **Cost Savings:** Reduce customer acquisition costs through targeted retention
- **Revenue Protection:** Focus resources on customers with highest lifetime value
- **Data-Driven Strategy:** Enable evidence-based customer management decisions

## Files in This Project
customer-churn-prediction/
├── README.md # This file
├── churn_data.csv # Dataset (7,043 records)
├── exploratory_analysis.ipynb # Complete analysis notebook
├── data_preprocessing.py # Data cleaning and preparation
├── model_training.py # Model training and evaluation
├── model_evaluation.py # Additional evaluation metrics
└── requirements.txt # Python dependencies
## How to Run

### Option 1: View Analysis Online
Open `exploratory_analysis.ipynb` directly in GitHub to see complete analysis with visualizations.

### Option 2: Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run preprocessing:**
```bash
python data_preprocessing.py
```

3. **Train models:**
```bash
python model_training.py
```

4. **Evaluate models:**
```bash
python model_evaluation.py
```

5. **View notebook:**
```bash
jupyter notebook exploratory_analysis.ipynb
```

## Key Insights

### Churn Drivers
- **Contract Type:** Customers with month-to-month contracts have higher churn risk
- **Tenure:** Longer tenure strongly correlates with retention
- **Monthly Charges:** Higher monthly charges are associated with increased churn
- **Services:** Customers using multiple services show lower churn rates

### Model Interpretation
- **Random Forest** outperforms Logistic Regression, suggesting non-linear relationships
- **Feature Importance:** Contract type, tenure, and monthly charges are top predictors
- **Threshold Optimization:** Can adjust prediction threshold based on business priorities

## Technologies Used

**Data & Analysis:**
- pandas: Data manipulation and analysis
- NumPy: Numerical computations
- scikit-learn: Machine learning models and metrics

**Visualization:**
- Matplotlib: Basic plotting
- Seaborn: Statistical data visualization

**Model Building:**
- Logistic Regression: Baseline classification
- Random Forest: Ensemble classification

## Author
**Navya Ravindran**
- Business Analyst & Data Professional
- 8+ years experience in analytics and BI
- Passionate about data-driven decision making

## Contact
📧 navyar91@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/navya-ravindran-762a6072)

---

## Next Steps
- Deploy model to production environment
- Monitor model performance and retrain as needed
- Implement retention strategies for high-risk customers
- Measure impact on actual churn rates and customer lifetime value
- Expand analysis to additional customer datasets

---

*Last Updated: July 31, 2026*
