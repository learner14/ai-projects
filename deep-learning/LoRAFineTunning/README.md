# LoRA Fine-Tuning Notebook (GPT-2 for SMS Spam Classification)

This project contains a single Jupyter notebook, `Implementing_LoRA_Fine_Tuning.ipynb`, that demonstrates how to fine-tune a GPT-2 model for **binary SMS spam classification** using **LoRA (Low-Rank Adaptation)** in PyTorch.

## What this notebook does

- Downloads and prepares the UCI SMS Spam Collection dataset.
- Creates train/validation/test CSV splits.
- Tokenizes text with GPT-2 tokenizer (`tiktoken`).
- Loads pretrained GPT-2 weights.
- Adapts GPT-2 for classification (2 output classes).
- Implements LoRA layers and injects them into linear layers.
- Trains only LoRA parameters (base model frozen).
- Evaluates train/validation/test accuracy and plots loss.

## Project structure

```
LoRAFineTunning/
├── Implementing_LoRA_Fine_Tuning.ipynb
└── README.md
```

After running the notebook, you should also see generated files such as:

- `train.csv`
- `validation.csv`
- `test.csv`
- downloaded dataset artifacts under `sms_spam_collection/` (and zip file)
- downloaded GPT-2 weights under `gpt2/`

## Requirements

- Python 3.10+ (3.11 recommended)
- Jupyter Notebook / VS Code Notebook kernel
- PyTorch
- pandas
- tiktoken
- matplotlib (for plotting, depending on your helper function implementation)

Install core packages:

```bash
pip install torch pandas tiktoken matplotlib
```

## Required helper modules

The notebook imports helper functions/modules that are **not inside this folder**:

- `gpt_download` (for downloading/loading GPT-2 weights)
- `previous_chapters` (for `GPTModel`, generation helpers, training loop, plotting)

Make sure these files/modules are available on your Python path (for example, copied into the same directory as the notebook, installed as a package, or added via `PYTHONPATH`).

## Important notebook note

In the data-download cell, `zipfile` and `os` are used. Ensure they are imported before running that cell:

```python
import os
import zipfile
```

## How to run

1. Open `Implementing_LoRA_Fine_Tuning.ipynb`.
2. Select a Python kernel with required packages installed.
3. Run cells top-to-bottom.
4. Verify:
   - dataset files are created,
   - GPT-2 weights download successfully,
   - train/validation loss plot is produced,
   - final train/validation/test accuracies print at the end.

## Expected workflow

1. **Data preparation**: download + balance ham/spam data.
2. **Dataset/Dataloader setup**: encode + pad SMS text.
3. **Model setup**: load GPT-2, replace output head for 2 classes.
4. **LoRA injection**: wrap linear layers with low-rank adapters.
5. **Training**: optimize LoRA parameters with AdamW.
6. **Evaluation**: compute accuracy on train/val/test splits.

## Troubleshooting

- **`ModuleNotFoundError: previous_chapters` or `gpt_download`**  
  Add these modules to your project path or place them next to the notebook.

- **Tokenizer/model download issues**  
  Check internet access and retry. The dataset cell already includes a backup URL.

- **Slow training on CPU**  
  Use CUDA (or MPS on Apple Silicon, if enabled in the notebook comments) for faster runs.

- **Out-of-memory errors**  
  Lower `batch_size` and/or use a smaller GPT-2 model.

## Credits

Notebook demonstrates LoRA-based parameter-efficient fine-tuning concepts for GPT-style models, adapted to SMS spam classification.