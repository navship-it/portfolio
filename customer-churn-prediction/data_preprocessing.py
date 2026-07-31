import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CUSTOMER CHURN PREDICTION - DATA PREPROCESSING")
print("=" * 60)

# Load data
print("\n📥 Loading dataset...")
df = pd.read_csv('churn_data.csv')

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Display basic info
print("\n📊 Dataset Overview:")
print(f"   First few rows:")
print(df.head())

print(f"\n   Data types:")
print(df.dtypes)

# Check missing values
print(f"\n🔍 Missing values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found")
else:
    print(missing[missing > 0])

# Handle missing values if any
df = df.dropna()

print(f"\n✅ After cleaning: {df.shape[0]} rows")

# Encode target variable
print("\n🔄 Encoding Churn variable...")
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])
print(f"   Churn values: {df['Churn'].unique()}")

# Check class distribution
print(f"\n📈 Churn distribution:")
churn_counts = df['Churn'].value_counts()
print(f"   No Churn (0): {churn_counts[0]} ({churn_counts[0]/len(df)*100:.1f}%)")
print(f"   Churn (1): {churn_counts[1]} ({churn_counts[1]/len(df)*100:.1f}%)")

# Separate features and target
print("\n🔀 Preparing features and target...")
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# Encode categorical variables
print("\n🔤 Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    print(f"   ✅ {col} encoded")

# Scale numerical features
print("\n📏 Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   ✅ Features scaled using StandardScaler")

# Train-test split
print("\n✂️  Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Train set: {X_train.shape[0]} samples")
print(f"   ✅ Test set: {X_test.shape[0]} samples")
print(f"   ✅ Train/Test split: 80/20")

# Save processed data
print("\n💾 Saving processed data...")
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print(f"   ✅ Data saved as .npy files")

print("\n" + "=" * 60)
print("✅ DATA PREPROCESSING COMPLETE!")
print("=" * 60)
print("\nNext step: Run model_training.py to train models")
