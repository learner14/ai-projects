# Instruction Fine-Tuning with GPT-2 (From Scratch in PyTorch)

This folder contains a single notebook, `Instruction-Fine-Tuning.ipynb`, that walks through **supervised instruction fine-tuning** of a GPT-2 model for instruction-following behavior.

The notebook builds most core components manually (dataset processing, collate function, GPT architecture blocks, training loop, text generation) and then fine-tunes on an instruction dataset.

---

## Contents

- [Overview](#overview)
- [What You Will Learn](#what-you-will-learn)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Notebook Pipeline](#notebook-pipeline)
- [How to Run](#how-to-run)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Known Caveats in the Notebook](#known-caveats-in-the-notebook)
- [Troubleshooting](#troubleshooting)
- [Customization Ideas](#customization-ideas)

---

## Overview

The notebook performs the following end-to-end workflow:

1. Downloads an instruction dataset in JSON format.
2. Formats each sample into an instruction-style prompt template.
3. Splits data into train/validation/test sets.
4. Tokenizes examples with GPT-2 tokenizer (`tiktoken`).
5. Builds custom collate functions with:
   - dynamic batch padding,
   - shifted targets for next-token prediction,
   - padding masking via `ignore_index=-100`.
6. Implements GPT model components in PyTorch:
   - multi-head causal self-attention,
   - transformer block,
   - feed-forward network,
   - layer normalization.
7. Downloads pretrained GPT-2 weights and maps them into the custom model.
8. Runs baseline generation, fine-tunes the model, and evaluates generated responses.
9. Plots training/validation loss.

---

## What You Will Learn

- How instruction tuning data is formatted for autoregressive language models.
- Why collate-time padding and label masking matter for stable training.
- How GPT internals (attention, feed-forward, residuals, norm) fit together.
- How to load pretrained GPT-2 weights into a custom PyTorch implementation.
- How to run a practical fine-tuning loop and inspect model outputs qualitatively.

---

## Project Structure

```text
instruction-fine-tunning/
├── Instruction-Fine-Tuning.ipynb
└── README.md
```

Generated at runtime (after executing notebook):

- `instruction-data.json` (downloaded dataset)
- `gpt2/` (downloaded pretrained weights)
- `loss-plot.pdf` (training curve)

---

## Environment Setup

### Recommended

- Python 3.10+ (3.11 recommended)
- Jupyter Notebook or VS Code Notebook kernel
- CUDA GPU if available (fastest)
- Apple Silicon MPS is also supported in notebook logic

### Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## Dependencies

Install core dependencies:

```bash
pip install torch tiktoken numpy matplotlib
```

The notebook also uses Python stdlib modules (`json`, `os`, `urllib`, `time`, `functools`).

### External helper module required

The notebook imports:

- `gpt_download` (`download_and_load_gpt2`)

Make sure this module is available in your Python path (same directory, installed package, or configured `PYTHONPATH`).

---

## Notebook Pipeline

### 1) Data download and formatting

- Downloads `instruction-data.json` from the `rasbt/LLMs-from-scratch` GitHub repo.
- Uses `format_input(entry)` to build prompt text:
  - Instruction header,
  - optional Input section,
  - expected Response section.

### 2) Train/val/test split

- Uses a fixed ratio:
  - **85% train**
  - **10% test**
  - **5% validation**

### 3) Dataset + tokenization

- `InstructionDataset` pre-tokenizes full prompt+response text with GPT-2 tokenizer.
- Tokenizer: `tiktoken.get_encoding("gpt2")`.

### 4) Collate function design

Three collate stages are shown:

- `custom_collate_draft_1`: pads inputs.
- `custom_collate_draft_2`: creates shifted inputs/targets.
- `custom_collate_fn`: production version with:
  - `ignore_index=-100` for padded target positions,
  - optional truncation with `allowed_max_length`.

### 5) Dataloaders

- `batch_size = 8`
- configurable device-aware collate using `functools.partial`
- train loader shuffles; val/test do not

### 6) GPT model implementation

Notebook defines:

- `GPTModel`
- `MultiHeadAttention`
- `LayerNorm`
- `GELU`
- `FeedForward`
- `TransformerBlock`

This is a clean educational implementation of a decoder-only transformer.

### 7) Pretrained weights loading

- Downloads GPT-2 medium (355M) weights.
- Uses `assign(...)` + `load_weights_into_gpt(...)` to map NumPy arrays into model parameters.

### 8) Generation and training

- Includes custom `generate(...)` with optional temperature/top-k.
- Computes initial train/val loss before training.
- Trains with `AdamW` (`lr=5e-5`, `weight_decay=0.1`) for `num_epochs=2`.
- Logs periodic train/val losses and prints sample generations.

### 9) Evaluation and plotting

- Plots training/validation loss and saves `loss-plot.pdf`.
- Generates responses for sample test entries and compares to reference outputs.

---

## How to Run

1. Open `Instruction-Fine-Tuning.ipynb`.
2. Select a Python kernel with required packages installed.
3. Ensure `gpt_download` module is importable.
4. Run cells sequentially from top to bottom.
5. Confirm expected artifacts appear (`instruction-data.json`, `gpt2/`, `loss-plot.pdf`).

If running for the first time, weight download may take time.

---

## Outputs and Artifacts

During/after execution, you should see:

- dataset size and sample entries printed,
- train/val/test split sizes,
- collate function sanity outputs,
- device printout (`cuda`, `mps`, or `cpu`),
- generation before and after fine-tuning,
- periodic training logs,
- final qualitative test examples,
- `loss-plot.pdf` saved to disk.

---

## Known Caveats in the Notebook

These are useful to address if you want smooth first-run execution:

1. **Missing plotting imports**
   - `plot_losses(...)` uses `plt` and `MaxNLocator`.
   - Add imports before plotting:

   ```python
   import matplotlib.pyplot as plt
   from matplotlib.ticker import MaxNLocator
   ```

2. **Notebook contains duplicate model setup blocks**
   - There are two places where base config/model download are created.
   - This is fine for learning, but you may streamline into one block.

3. **Large model by default**
   - GPT-2 medium (355M) may be slow/heavy on CPU.
   - Switch to `gpt2-small (124M)` for faster experimentation.

---

## Troubleshooting

- **`ModuleNotFoundError: gpt_download`**
  - Add the helper module to this folder or install/package it properly.

- **Out-of-memory / very slow training**
  - Use smaller model, reduce `batch_size`, or reduce `allowed_max_length`.

- **No GPU detected**
  - Training runs on CPU; expect slower runtime.

- **Network/download failure**
  - Retry dataset and GPT-2 download cells once internet is stable.

- **Plotting errors (`plt` or `MaxNLocator` undefined)**
  - Add the plotting imports in the caveats section above.

---

## Customization Ideas

- Try different GPT sizes (`small`, `medium`, `large`).
- Tune learning rate, batch size, and epochs.
- Add checkpoint saving per epoch.
- Add quantitative generation metrics (BLEU/ROUGE or exact-match style task metrics).
- Add train/val shuffling seeds and experiment tracking for reproducibility.

---

If you want, the next improvement is to add a short **Quick Start** markdown cell at the top of the notebook with required imports, expected runtime, and a one-screen run checklist.