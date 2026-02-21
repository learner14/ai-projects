# Performance Measures on MNIST (Binary Classification: Digit `5` vs Not `5`)

This project demonstrates how to evaluate a machine learning classifier using core **performance measures** in a practical setting.

The notebook trains a binary classifier on the MNIST dataset to answer a single question:

- **Is this image the digit `5`?**

It then evaluates the classifier using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Precision-Recall vs Threshold behavior
- ROC Curve
- ROC-AUC

---

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `Performance_Measures.ipynb` in VS Code or Jupyter.
4. Run all cells from top to bottom.

---

## Project Structure

- `Performance_Measures.ipynb` — main notebook containing all code, plots, and evaluation steps.
- `requirements.txt` — Python dependencies needed to run the notebook.

---

## Goal of the Notebook

Accuracy alone can be misleading in imbalanced or asymmetric problems. This notebook shows **why multiple metrics are needed** and how threshold tuning affects model behavior.

For the MNIST case:

- Positive class = images of digit `5`
- Negative class = images of all other digits

---

## Libraries Used

The notebook uses:

- `scikit-learn` (`fetch_openml`, `SGDClassifier`, model selection and metrics)
- `matplotlib` (visualization)

Primary imports in the notebook:

- `from sklearn.datasets import fetch_openml`
- `from sklearn.linear_model import SGDClassifier`
- `from sklearn.model_selection import cross_val_score, cross_val_predict`
- `from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score`
- `from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score`

---

## Dataset

Dataset: **MNIST (`mnist_784`)** from OpenML.

- Total samples: 70,000
- Feature vector size: 784 (28×28 flattened grayscale image)

Split strategy used in notebook:

- Training set: first 60,000 samples
- Test set: last 10,000 samples

Target transformation:

- `y_train_5 = (y_train == '5')`
- `y_test_5 = (y_test == '5')`

This converts the multiclass problem into binary classification.

---

## Model

Model used: `SGDClassifier(random_state=42)`

- Trained on `X_train` and binary labels `y_train_5`
- Produces decision scores via `decision_function`
- These scores are later used for threshold analysis, precision-recall curve, and ROC curve

---

## Notebook Workflow (Step-by-Step)

1. Import libraries.
2. Download/load MNIST.
3. Inspect shape and visualize an example digit.
4. Create train/test split.
5. Create binary target labels (`5` vs not `5`).
6. Train `SGDClassifier`.
7. Run baseline prediction on one sample.
8. Evaluate with cross-validation accuracy.
9. Get out-of-fold predictions using `cross_val_predict`.
10. Build confusion matrix.
11. Compute precision, recall, and F1.
12. Compute decision scores.
13. Build precision-recall curve and inspect threshold trade-offs.
14. Build ROC curve and locate threshold point for ~90% precision.
15. Compute ROC-AUC.

---

## Saved Example Results (from notebook output)

### 1) Cross-Validation Accuracy (`cv=3`)

`[0.95035, 0.96035, 0.9604]`

Interpretation:

- Accuracy appears high (~95% to 96%), but this does not reveal class-specific errors.

### 2) Confusion Matrix

```
[[53892,   687],
 [ 1891,  3530]]
```

Interpreting matrix layout:

- TN = 53,892
- FP = 687
- FN = 1,891
- TP = 3,530

Observation:

- False negatives are relatively high for the positive class (`5`), affecting recall.

### 3) Precision

`0.8370879772350012`

Meaning:

- Of all samples predicted as `5`, ~83.7% are actually `5`.

### 4) Recall

`0.6511713705958311`

Meaning:

- Of all true `5`s, only ~65.1% are captured by the classifier.

### 5) F1-Score

`0.7325171197343847`

Meaning:

- Harmonic balance between precision and recall.

### 6) Threshold for ~90% Precision

`3370.0194991439557`

Meaning:

- Increasing threshold generally increases precision but decreases recall.

### 7) ROC-AUC

`0.9604938554008616`

Meaning:

- Strong ranking performance across thresholds.

---

## Metric Definitions (Quick Reference)

Let:

- TP = true positives
- TN = true negatives
- FP = false positives
- FN = false negatives

Formulas:

- **Accuracy** = `(TP + TN) / (TP + TN + FP + FN)`
- **Precision** = `TP / (TP + FP)`
- **Recall** = `TP / (TP + FN)`
- **F1-score** = `2 * (Precision * Recall) / (Precision + Recall)`
- **ROC-AUC** = area under ROC curve (TPR vs FPR over all thresholds)

---

## Why This Notebook Is Useful

This notebook is a compact and practical reference for:

- Understanding why accuracy is not enough
- Reading and interpreting confusion matrices
- Balancing precision and recall with threshold tuning
- Visualizing model discrimination with ROC and PR concepts

It is especially useful for classification tasks where false positives and false negatives carry different costs.

---

## How to Run

### Option 1: In VS Code (recommended)

1. Open `Performance_Measures.ipynb`.
2. Select a Python kernel.
3. Run cells top to bottom.

### Option 2: Jupyter Notebook/Lab

1. Install dependencies:
   ```bash
   pip install scikit-learn matplotlib jupyter
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open and run `Performance_Measures.ipynb`.

---

## Notes

- The notebook uses OpenML download for MNIST (`fetch_openml`), so internet access is needed on first run.
- Some cells in the notebook are placeholders/empty; this is normal and does not affect the main flow.
- If you rerun with different random states or preprocessing, metric values may shift.

---

## Suggested Next Improvements

- Compare `SGDClassifier` with a stronger baseline (e.g., Random Forest or SVM).
- Add threshold selection based on business objective (maximize recall at minimum precision, etc.).
- Evaluate on the held-out test set using the chosen threshold.
- Add calibration analysis if probability quality matters.
