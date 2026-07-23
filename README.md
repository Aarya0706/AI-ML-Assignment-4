# Assignment 4 – Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
A healthcare organization wants to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements. This project builds a **K-Nearest Neighbors (KNN)** classification model (K = 5) to classify tumors accurately.

## Dataset Link
Breast Cancer Wisconsin (Diagnostic) Dataset (Kaggle):
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> The dataset is **not** included in this repository, per the assignment instructions. Download the CSV from the Kaggle link above (or use the equivalent `sklearn.datasets.load_breast_cancer` data) and place it as `breast_cancer_data.csv` in the project root before running the code.

## Libraries Used
- `pandas` – data loading and manipulation
- `numpy` – numerical operations
- `matplotlib` – data visualization
- `scikit-learn` – `StandardScaler`, `LabelEncoder`, `train_test_split`, `KNeighborsClassifier`, evaluation metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `ConfusionMatrixDisplay`)

## Methodology
1. **Data Understanding** – Loaded the dataset with Pandas, inspected the first five records, identified the 30 numerical diagnostic measurements as features and `diagnosis` as the target variable, and reviewed dataset info/summary statistics.
2. **Data Preprocessing** – Checked for missing values, dropped the unnecessary `id` column and the empty `Unnamed: 32` column, label-encoded `diagnosis` (M → 1, B → 0), standardized all numerical features with `StandardScaler` (essential for a distance-based algorithm like KNN), and split the data into 80% training / 20% testing (stratified on the target).
3. **Model Development** – Trained a `KNeighborsClassifier` with **K = 5** on the scaled training data and predicted class labels on the test set.
4. **Model Evaluation** – Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and visualized performance with a confusion matrix.

## Results
| Metric | Value |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9744 |
| Recall | 0.9048 |
| F1-Score | 0.9383 |

**Confusion Matrix:**
```
[[71  1]
 [ 4 38]]
```
*(Rows/columns ordered as [B, M]; see `confusion_matrix.png` for the plotted version.)*

**Observations:**
1. The KNN model (K=5) achieves high accuracy, showing the diagnostic measurements provide strong signal to separate malignant from benign tumors once features are scaled.
2. The confusion matrix shows very few misclassifications; false negatives (predicting benign when actually malignant) matter far more in a medical context than false positives, so recall on the malignant class is worth monitoring closely.
3. Precision, Recall, and F1-Score are all close to accuracy, indicating consistent performance across both classes rather than a bias toward the majority class.

Generated plot:
- `confusion_matrix.png` – confusion matrix for the KNN model (K=5)

## Conclusion
A K-Nearest Neighbors (KNN) classifier with K = 5 was built to classify breast tumors as Malignant or Benign using the Breast Cancer Wisconsin Diagnostic dataset. After removing the unnecessary `id` and empty `Unnamed: 32` columns, encoding the diagnosis label, and standardizing all 30 numerical features, the model achieved strong accuracy, precision, recall, and F1-scores on the held-out test set, with very few misclassifications in the confusion matrix.

Feature scaling is critical for KNN because the algorithm classifies a new point based on the **distance** to its nearest neighbors; features measured on larger scales (e.g., `area_mean`) would otherwise dominate the distance calculation and drown out equally important features measured on smaller scales (e.g., `smoothness_mean`), leading to biased predictions.

One key limitation of KNN is that it is **computationally expensive at prediction time** for large datasets, since it must calculate the distance from a new sample to every point in the training set — making it slow to scale compared to models that learn a fixed set of parameters during training.

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn
python Assignment-4.py
# or open Assignment-4.ipynb in Jupyter
```
