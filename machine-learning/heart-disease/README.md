# Heart Disease Prediction with XGBoost

This project uses **XGBoost (Extreme Gradient Boosting)** to predict whether a patient has heart disease using the notebook:

- `Xgboost_heart_disease.ipynb`

---

## What is XGBoost?

**XGBoost** is a high-performance implementation of gradient boosted decision trees. Instead of training a single tree, it builds many trees sequentially, where each new tree focuses on correcting the errors made by previous trees.

Why it is commonly used:
- Strong performance on tabular datasets
- Built-in regularization to reduce overfitting
- Efficient and scalable implementation
- Handles missing values with learned default directions in trees

In this notebook, XGBoost is used for **binary classification**:
- `0` = No heart disease
- `1` = Has heart disease

---

## Dataset Source (Cited)

The notebook states that the dataset comes from the **UCI Machine Learning Repository**:

- UCI ML Repository: https://archive.ics.uci.edu/ml/index.php
- Heart Disease Dataset page: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

The notebook reads the local file:
- `processed.cleveland.data`

This file corresponds to the Cleveland subset of the UCI Heart Disease dataset.

---

## Full Notebook Flow

The notebook follows this end-to-end workflow.

### 1) Import libraries
Main libraries used:
- `pandas`, `numpy`
- `xgboost`
- `sklearn.model_selection` (`train_test_split`, `GridSearchCV`)
- `sklearn.metrics` (`balanced_accuracy_score`, `roc_auc_score`, `ConfusionMatrixDisplay`)

### 2) Load data
- Loads data from `processed.cleveland.data` with `pd.read_csv(..., header=None)`
- Renames columns to meaningful feature names:
  - `age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, hd`

### 3) Identify missing values
- Inspects datatypes and unique values
- Finds `?` placeholders in `ca` and `thal`
- Counts rows with missing markers

### 4) Handle missing values (XGBoost-style)
- Replaces missing `?` in `ca` and `thal` with `0`
- Verifies replacement using `unique()`

### 5) Split features and target
- Creates feature matrix `X` by dropping `hd`
- Creates target `y` as `hd`

### 6) One-hot encode categorical variables
- Uses `pd.get_dummies()` to encode:
  - `cp`, `restecg`, `slope`, `thal`
- Keeps binary columns (`sex`, `fbs`, `exang`) as they are

### 7) Convert target to binary label
- Original target has 5 levels (`0, 1, 2, 3, 4`)
- Converts all values `> 0` to `1` for binary heart-disease prediction

### 8) Ensure numeric/boolean dtypes
- Converts `ca` to numeric using `pd.to_numeric()`
- Confirms all model inputs are valid dtypes for XGBoost

### 9) Train a preliminary XGBoost classifier
- Splits into train/test with `train_test_split(..., random_state=42)`
- Trains `xgb.XGBClassifier(objective='binary:logistic', seed=42)`
- Evaluates with confusion matrix on test data

### 10) Hyperparameter tuning with GridSearchCV
- Grid-searches over:
  - `max_depth`
  - `n_estimators`
  - `learning_rate`
  - `gamma`
  - `reg_lambda`
- Uses 5-fold cross-validation
- Best values documented in notebook text: `gamma=1`, `learning_rate=0.1`, `max_depth=3`, `n_estimators=200` (and regularization setting)

### 11) Train optimized model and evaluate
- Fits tuned `XGBClassifier`
- Compares confusion matrix vs preliminary model
- Shows modest changes in class-wise performance

### 12) Interpret model/tree behavior
- Trains a 1-tree version (`n_estimators=1`) for interpretability
- Prints feature importance metrics:
  - `weight`, `gain`, `cover`, `total_gain`, `total_cover`
- Visualizes tree with `xgb.to_graphviz(...)`

---

## Project Outcome

By the end of the notebook, the pipeline demonstrates:
- Data cleaning and formatting for XGBoost
- Binary heart disease classification
- Baseline vs tuned model comparison
- Basic model interpretability through feature importance and tree visualization

---

## Results Summary

Below is a compact comparison of the confusion-matrix performance reported in the notebook narrative.

| Model | No HD classified correctly | Has HD classified correctly |
|---|---:|---:|
| Preliminary XGBoost | 31 / 39 (79%) | 30 / 37 (81%) |
| Optimized XGBoost | 35 / 39 (90%) | 31 / 37 (85%) |

Key takeaway:
- Tuning improved identification of patients **without** heart disease.
- Detection of patients **with** heart disease stayed similar, with a slight decrease.

---

## How to run

1. Place `processed.cleveland.data` in the same directory as the notebook.
2. Open `Xgboost_heart_disease.ipynb`.
3. Run cells from top to bottom.

> Note: The grid search section can take a few minutes depending on machine resources.
