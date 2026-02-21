# ResNet-34 from Scratch in PyTorch (CIFAR-10)

This folder contains [Resnet.ipynb](Resnet.ipynb), a notebook that builds and trains a custom **ResNet-34** for image classification using PyTorch.

It covers:
- residual block design,
- full ResNet-34 architecture assembly,
- CIFAR-10 data loading and normalization,
- training/validation/testing loops,
- metric plotting and run summary.

---

## Project structure

```text
ResNet-model/
├── Resnet.ipynb
└── README.md
```

Expected artifacts/output from notebook execution:
- CIFAR-10 dataset files under `./datasets`
- console logs for epoch-wise train/validation metrics
- plots for loss/accuracy curves
- optional model checkpoint file (`my_resnet34_checkpoint.pt`)

---

## How ResNet works (conceptual)

Traditional very-deep CNNs can be hard to optimize due to degraded gradient flow. ResNet solves this with **skip (identity) connections**.

Instead of learning a direct mapping $H(x)$, each block learns a residual mapping $F(x)$ and outputs:

$$
y = F(x) + x
$$

Then an activation is applied (ReLU in this notebook).

### Why this helps

- Easier optimization of deeper models.
- Better gradient propagation through skip paths.
- Strong performance with scalable depth.

---

## Residual unit used in this notebook

The custom `ResidualUnit` includes:

- Main path:
  - Conv(3x3) -> BN -> ReLU
  - Conv(3x3) -> BN
- Skip path:
  - Identity when shape is unchanged (`stride=1`)
  - 1x1 projection + BN when downsampling (`stride>1`)
- Merge:
  - element-wise add main + skip
  - final ReLU

This is the core building block repeated across stages.

---

## ResNet-34 architecture implemented here

The notebook builds:

1. **Stem**
   - `Conv7x7, stride=2`
   - `BatchNorm`, `ReLU`
   - `MaxPool3x3, stride=2`

2. **Residual stages**
   - Stage 1: 3 blocks @ 64 channels
   - Stage 2: 4 blocks @ 128 channels (first block downsamples)
   - Stage 3: 6 blocks @ 256 channels (first block downsamples)
   - Stage 4: 3 blocks @ 512 channels (first block downsamples)

3. **Classifier head**
   - `AdaptiveAvgPool2d(1)`
   - `Flatten`
   - `Linear -> 10 classes` (`nn.LazyLinear(10)`)

This corresponds to a classic ResNet-34 style configuration.

---

## Notebook workflow (section-by-section)

### 1) Imports and setup
Loads PyTorch, torchvision, transforms, plotting, and utility modules.

### 2) Define `ResidualUnit`
Implements the residual building block with projection skip on downsampling.

### 3) Define `ResNet34`
Constructs the full network by stacking residual units with stage-wise channel progression.

### 4) Load CIFAR-10 + dataloaders
- normalization is applied,
- train split is further divided into train (45k) and validation (5k),
- dataloaders are created for train/val/test.

### 5) Visual inspection
Displays sample images (denormalized for readability).

### 6) Training setup
- device selection (`mps` if available else CPU),
- `CrossEntropyLoss`,
- Adam optimizer (`lr=1e-3`).

### 7) Train and evaluate loops
Defines:
- `train_epoch(...)`
- `test_epoch(...)`

Then runs 10 epochs and logs metrics per epoch.

### 8) Final test + plots
Evaluates on test set and visualizes train/val/test trends.

---

## Environment setup

Recommended:
- Python 3.10+
- PyTorch + torchvision
- matplotlib
- scikit-learn (imported in notebook)

Install example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib scikit-learn torchmetrics
```

---

## How to run

1. Open [ResNet-model/Resnet.ipynb](ResNet-model/Resnet.ipynb).
2. Select a Python kernel with dependencies installed.
3. Ensure CIFAR-10 is available locally or set `download=True` in dataset cells.
4. Run cells in order from top to bottom.
5. Confirm:
   - epoch logs print,
   - test metrics are produced,
   - plots render correctly.

---

## Current caveats in this notebook

1. **CIFAR-10 download disabled by default**
   - Dataset cells use `download=False`.
   - If dataset is not already present, set to `download=True`.

2. **Plot cell uses `test_loss` / `test_acc` before assignment**
   - The plotting cell appears before test evaluation cell.
   - Run test evaluation first or move plotting cell after it.

3. **Model save line is effectively commented out**
   - There is a typo in the final save comment (`# Save the trained modeltorch.save(...)`).
   - Add a proper save line if you want checkpoint output:

```python
torch.save(model.state_dict(), './my_resnet34_checkpoint.pt')
```

---

## Expected outcomes

After a successful run:
- model learns progressively across epochs,
- validation and test accuracy are printed,
- overfitting can be inspected via train/val/test gap,
- plots summarize optimization behavior.

---

## Suggested next improvements

- Add learning-rate scheduler (`CosineAnnealingLR` or `StepLR`).
- Add data augmentation (random crop/flip).
- Save best checkpoint by validation accuracy.
- Track precision/recall/F1 in addition to accuracy.
- Compare against pretrained `torchvision.models.resnet34` baseline.

---

If you want, I can also patch [ResNet-model/Resnet.ipynb](ResNet-model/Resnet.ipynb) with minimal fixes (dataset download flag, plot/test ordering, and checkpoint save line) so it runs smoothly end to end.