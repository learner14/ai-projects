# MyGPT2Model: Building GPT-2 from Scratch (and Loading Pretrained Weights)

This project contains `MyGPT2Model.ipynb`, an educational end-to-end notebook that:

1. implements core GPT-2 architecture components in PyTorch,
2. trains a small GPT-style model on a tiny text corpus,
3. demonstrates decoding strategies,
4. saves/loads checkpoints,
5. loads official OpenAI GPT-2 weights into the custom implementation.

If your goal is to understand **how GPT-2 works internally**, this notebook is a practical, code-first walkthrough.

---

## Project structure

```text
my-gpt2-model/
├── MyGPT2Model.ipynb
└── README.md
```

Artifacts produced during execution:
- `the-verdict.txt` (training text corpus)
- `loss-plot.pdf` (train/val loss curve)
- `model_and_optimizer.pth` (checkpoint)
- `gpt2/` (downloaded pretrained model files)

---

## GPT-2 in plain language

GPT-2 is a **decoder-only Transformer** trained with a next-token prediction objective.

Given a token sequence:

- it embeds tokens + positions,
- passes them through stacked Transformer blocks,
- predicts a probability distribution over the vocabulary for each position,
- is trained to predict the next token (targets are shifted inputs).

At inference, it generates text autoregressively (one token at a time), feeding each new token back into the model.

### Core components

1. **Token embedding (`tok_emb`)**
   - Maps token IDs to dense vectors.

2. **Positional embedding (`pos_emb`)**
   - Adds position information (order awareness).

3. **Transformer block (repeated `n_layers`)**
   - Causal multi-head self-attention
   - Feed-forward network
   - Residual connections + layer normalization

4. **Output head (`out_head`)**
   - Projects hidden states to vocabulary logits.

5. **Causal mask**
   - Prevents attending to future tokens during training/inference.

---

## How this notebook maps to GPT-2 concepts

### 1) Environment and imports
The notebook checks package versions and imports `torch`, `tiktoken`, and utility modules.

### 2) GPT-2 architecture implementation
You define from scratch:

- `MultiHeadAttention`
- `LayerNorm`, `GELU`, `FeedForward`
- `TransformerBlock`
- `GPTModel`

This mirrors the decoder stack used by GPT-2.

### 3) Minimal text generation
`generate_text_simple(...)` performs greedy decoding:
- take last-token logits,
- pick `argmax`,
- append token,
- repeat.

### 4) Loss and perplexity foundations
The notebook demonstrates:
- logits/target shapes,
- flattening batch+time dimensions,
- `cross_entropy` for next-token prediction.

### 5) Data pipeline for language modeling
- downloads a small corpus (`the-verdict.txt`),
- tokenizes with GPT-2 tokenizer,
- builds shifted input-target pairs via `GPTDatasetV1`,
- creates train/validation dataloaders.

### 6) Training loop
`train_model_simple(...)` includes:
- batch loss computation,
- optimizer update (`AdamW`),
- periodic train/val evaluation,
- sample generation after each epoch.

### 7) Decoding controls
`generate(...)` extends greedy decoding with:
- **temperature** scaling,
- **top-k** filtering,
- optional `eos_id` stopping.

### 8) Save and restore
Shows checkpointing with:
- `torch.save({...})`
- `torch.load(...)`
- reloading model and optimizer state.

### 9) Loading official GPT-2 pretrained weights
- uses `gpt_download.download_and_load_gpt2(...)`,
- creates GPT-2 config matching OpenAI dimensions,
- maps downloaded NumPy weights into the custom PyTorch model via `assign(...)` and `load_weights_into_gpt(...)`,
- generates text from the pretrained-loaded model.

---

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- tiktoken
- TensorFlow (used by the helper downloader in this workflow)
- tqdm

Install packages:

```bash
pip install torch numpy matplotlib tiktoken tensorflow tqdm
```

---

## Required helper module

The notebook imports:

- `from gpt_download import download_and_load_gpt2`

Ensure `gpt_download.py` is available on your Python path (same folder, installed module, or configured `PYTHONPATH`).

---

## How to run

1. Open `MyGPT2Model.ipynb`.
2. Select the Python kernel with required packages.
3. Run cells top to bottom.
4. Validate outputs:
   - generated sample text before/after training,
   - printed train/val losses,
   - saved `loss-plot.pdf`,
   - successful pretrained GPT-2 loading and generation.

---

## Configuration notes

The notebook uses two practical configurations:

1. **Small educational GPT config**
   - context length reduced to 256 for faster local experiments.

2. **Pretrained-compatible config**
   - sets context length to 1024 and `qkv_bias=True` so OpenAI GPT-2 weights map correctly.

---

## Known caveats

1. **Tiny training corpus overfits quickly**
   - This is expected and educational.

2. **`pip install` inside a code cell**
   - Notebook includes `pip install tensorflow tqdm` in a cell.
   - In some environments, prefer installing in terminal before running notebook.

3. **`gpt_download` dependency is external to this folder**
   - If missing, pretrained-weight loading section will fail.

4. **Device choice**
   - Default is CUDA if available, otherwise CPU.
   - MPS path is commented and can be enabled on Apple Silicon.

---

## Why this notebook is useful

It bridges the gap between “using GPT-2” and “understanding GPT-2 implementation details.”

You get:
- conceptual understanding (attention, masking, next-token loss),
- implementation-level familiarity,
- practical training/inference skills,
- compatibility with official pretrained GPT-2 weights.

---

## Suggested next steps

- Add validation perplexity tracking.
- Add text generation evaluation prompts set.
- Compare greedy vs top-k/temperature outputs systematically.
- Add checkpoint naming by epoch/step.
- Extend to instruction fine-tuning or LoRA adaptation.

---

If you want, I can also add a short **Quick Start** section at the top of the notebook (Cell 1 markdown) with expected runtime, GPU notes, and a run checklist.