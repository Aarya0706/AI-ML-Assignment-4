"""
Assignment 4: Breast Cancer Classification using K-Nearest Neighbors (KNN)
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (Kaggle)
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------
df = pd.read_csv('breast_cancer_data.csv')

print("First five records:")
print(df.head(), "\n")

print("Numerical Features: 30 diagnostic measurement columns "
      "(radius_mean, texture_mean, ... _se, ... _worst)")
print("Target Variable: diagnosis (M = Malignant, B = Benign)\n")

print("Dataset info:")
df.info()
print("\nSummary statistics:")
print(df.describe(), "\n")

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# Remove unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode the target variable
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M -> 1, B -> 0
print("Classes:", list(le.classes_))
print(df['diagnosis'].value_counts(), "\n")

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Standardize the feature values (important for distance-based KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into 80% training / 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n")

# --------------------------------------------------------------------------
# Task 3: Model Development
# --------------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - KNN (K=5)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

print("\nPlot saved: confusion_matrix.png")

# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------
print("""
Observations:
1. The KNN model (K=5) achieves high accuracy, showing the diagnostic
   measurements provide strong signal to separate malignant from benign
   tumors once features are scaled.
2. The confusion matrix shows very few misclassifications; false negatives
   (predicting benign when actually malignant) matter far more in a medical
   context than false positives.
3. Precision, Recall, and F1-Score are all close to accuracy, indicating
   consistent performance across both classes rather than a bias toward
   the majority class.
""")

# --------------------------------------------------------------------------
# Task 5: Conclusion
# --------------------------------------------------------------------------
print("""
Conclusion:
A K-Nearest Neighbors (KNN) classifier with K=5 was built to classify breast
tumors as Malignant or Benign using the Breast Cancer Wisconsin Diagnostic
dataset. After removing the unnecessary 'id' and empty 'Unnamed: 32' columns,
encoding the diagnosis label, and standardizing all 30 numerical features,
the model achieved strong accuracy, precision, recall, and F1-scores on the
held-out test set, with very few misclassifications in the confusion matrix.

Feature scaling is critical for KNN because the algorithm classifies a new
point based on the distance to its nearest neighbors; features on larger
scales (e.g., area_mean) would otherwise dominate the distance calculation
and drown out equally important features on smaller scales (e.g.,
smoothness_mean), leading to biased predictions.

One key limitation of KNN is that it is computationally expensive at
prediction time for large datasets, since it must calculate the distance
from a new sample to every point in the training set -- making it slow to
scale compared to models that learn a fixed set of parameters during
training.
""")
