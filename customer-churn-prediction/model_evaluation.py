import numpy as np
import pickle
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    classification_report, roc_auc_score
)
import matplotlib.pyplot as plt

print("=" * 60)
print("MODEL EVALUATION & VISUALIZATION")
print("=" * 60)

# Load data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load model
with open('best_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("\n📊 MODEL PERFORMANCE:")
print(f"   Accuracy: {(y_pred == y_test).sum() / len(y_test):.4f}")
print(f"   ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔲 Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"   True Negatives: {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives: {tp}")

# Calculate rates
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\n📈 Sensitivity (Recall): {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")

print("\n✅ Evaluation complete!")
