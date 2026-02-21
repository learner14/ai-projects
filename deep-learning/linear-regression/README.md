# Linear Regression with PyTorch (California Housing)

This folder contains [linear-regression/LinearRegressionPytorch.ipynb](linear-regression/LinearRegressionPytorch.ipynb), a notebook that implements **linear regression from scratch using PyTorch tensors and autograd**.

The notebook uses the **California Housing** dataset to predict house values from tabular features.

---

## Project structure

```text
linear-regression/
├── LinearRegressionPytorch.ipynb
└── README.md
```

---

## What linear regression is doing

Linear regression learns a linear mapping from input features $X$ to target $y$:

$$
\hat{y} = Xw + b
$$

Where:
- $X$ is the feature matrix,
- $w$ is the weight vector,
- $b$ is the bias,
- $\hat{y}$ is the prediction.

The notebook optimizes $w$ and $b$ by minimizing **mean squared error (MSE)**:

$$
\mathcal{L}(w,b) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

Gradients are computed with PyTorch autograd, and parameters are updated using manual gradient descent.

---

## What this notebook builds (step-by-step)

### 1) Imports and environment checks
- Imports PyTorch, NumPy, Pandas, Matplotlib, and scikit-learn utilities.
- Prints library versions.

### 2) Dataset download/load
- Fetches California Housing via `fetch_california_housing()`.
- Separates:
  - `X` features (8 columns)
  - `y` target house value.

### 3) Data splitting + preprocessing
- Splits data into:
  - train (60%)
  - validation (20%)
  - test (20%)
- Converts arrays to `torch.FloatTensor`.
- Standardizes features using **train-set mean/std only**, then applies to val/test.

### 4) Model definition (manual parameters)
- Initializes:
  - `w` with shape `(num_features, 1)` and `requires_grad=True`
  - `b` scalar with `requires_grad=True`
- This is equivalent to a one-layer linear model.

### 5) Training loop
- Runs for `num_epochs = 1000` with `learning_rate = 0.04`.
- Per epoch:
  1. Forward: `y_pred = X_train @ w + b`
  2. Loss: MSE
  3. Backward: `loss.backward()`
  4. Parameter update in `torch.no_grad()`
  5. Zero gradients
- Logs training loss every 100 epochs.

### 6) Inference preview
- Predicts first 5 test samples.
- Prints predicted vs actual values.

---

## Why normalization matters here

Features in California Housing have different numeric ranges. Without scaling, some dimensions dominate gradients and training can be unstable/slower. Standardization makes optimization smoother and often converges faster.

---

## Environment setup

Recommended:
- Python 3.10+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas scikit-learn matplotlib
```

---

## How to run

1. Open [linear-regression/LinearRegressionPytorch.ipynb](linear-regression/LinearRegressionPytorch.ipynb).
2. Select a Python kernel with dependencies installed.
3. Run cells from top to bottom.
4. Confirm:
   - dataset loads,
   - train/val/test split sizes print,
   - training loss decreases over epochs,
   - prediction samples print at the end.

---

## Current notebook caveats

1. **Typo in import line**
   - Current cell has: `from sklearn.model_selection import train_test_splitå`
   - It should be:
   ```python
   from sklearn.model_selection import train_test_split
   ```

2. **No validation metric logging during training**
   - Validation set is created but not used in the loop.
   - You can add periodic val MSE to monitor generalization.

3. **No final quantitative test metric shown**
   - Notebook prints sample predictions, but no overall test MSE/MAE/R².

---

## Suggested next improvements

- Add validation and test MSE/MAE/R² calculations.
- Plot train/validation loss curves.
- Compare manual implementation vs `nn.Linear` + `optim.SGD`.
- Add mini-batch training with `DataLoader`.
- Save/load trained parameters (`w`, `b`).

---

If you want, I can also patch [linear-regression/LinearRegressionPytorch.ipynb](linear-regression/LinearRegressionPytorch.ipynb) to fix the import typo and add validation/test metrics in a minimal way.