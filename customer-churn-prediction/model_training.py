import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING & EVALUATION")
print("=" * 70)

# Load preprocessed data
print("\n📥 Loading preprocessed data...")
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"✅ Data loaded successfully!")
print(f"   Train set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Dictionary to store results
results = {}

# ============================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================
print("\n" + "=" * 70)
print("🤖 MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

print("\n⏳ Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("✅ Training complete!")

# Predictions
lr_pred = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# Metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred_proba)

results['Logistic Regression'] = {
    'accuracy': lr_accuracy,
    'precision': lr_precision,
    'recall': lr_recall,
    'f1': lr_f1,
    'auc': lr_auc
}

print(f"\n📊 Logistic Regression Results:")
print(f"   Accuracy:  {lr_accuracy:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall:    {lr_recall:.4f}")
print(f"   F1-Score:  {lr_f1:.4f}")
print(f"   ROC-AUC:   {lr_auc:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))

print(f"\n🔲 Confusion Matrix:")
cm_lr = confusion_matrix(y_test, lr_pred)
print(cm_lr)

# ============================================
# MODEL 2: RANDOM FOREST
# ============================================
print("\n" + "=" * 70)
print("🤖 MODEL 2: RANDOM FOREST")
print("=" * 70)

print("\n⏳ Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("✅ Training complete!")

# Predictions
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1': rf_f1,
    'auc': rf_auc
}

print(f"\n📊 Random Forest Results:")
print(f"   Accuracy:  {rf_accuracy:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall:    {rf_recall:.4f}")
print(f"   F1-Score:  {rf_f1:.4f}")
print(f"   ROC-AUC:   {rf_auc:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Churn', 'Churn']))

print(f"\n🔲 Confusion Matrix:")
cm_rf = confusion_matrix(y_test, rf_pred)
print(cm_rf)

# ============================================
# MODEL COMPARISON
# ============================================
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON")
print("=" * 70)

print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-" * 80)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = rf_model if best_model_name == 'Random Forest' else lr_model

print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['auc']:.4f})")

# Save best model
print(f"\n💾 Saving best model...")
with open('best_churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Model saved as 'best_churn_model.pkl'")

# Feature importance (Random Forest)
print(f"\n" + "=" * 70)
print("📈 FEATURE IMPORTANCE (Random Forest)")
print("=" * 70)

feature_importance = rf_model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:][::-1]

print(f"\nTop 10 Most Important Features:")
for i, idx in enumerate(top_features_idx, 1):
    print(f"   {i}. Feature {idx}: {feature_importance[idx]:.4f}")

print("\n" + "=" * 70)
print("✅ MODEL TRAINING & EVALUATION COMPLETE!")
print("=" * 70)
print("\nModels trained and saved successfully!")
print("You can now use these models for prediction on new data.")
