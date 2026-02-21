# Transformer from Scratch + NMT Pipeline (English ↔ Spanish)

This folder contains [transformer/Transformer.ipynb](transformer/Transformer.ipynb), a notebook that explains and implements key Transformer components for neural machine translation (NMT).

The notebook combines:
- custom Transformer building blocks (attention, encoder, decoder),
- tokenization and batching for sequence-to-sequence text,
- a translation model wrapper (`NmtTransformer`),
- training and evaluation utilities.

---

## Project structure

```text
transformer/
├── Transformer.ipynb
└── README.md
```

---

## How Transformers work (quick conceptual guide)

A Transformer performs sequence modeling using attention instead of recurrence.

### 1) Embedding + positional information
Tokens are mapped to dense vectors and enriched with position information.

If token embeddings are $E$ and positional embeddings are $P$, input is:
$$
X = E + P
$$

### 2) Multi-head attention
Each head computes scaled dot-product attention:
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Multi-head attention runs several heads in parallel, concatenates outputs, and projects them.

### 3) Encoder layer
Typical encoder block:
1. Self-attention + residual + layer norm
2. Feed-forward network + residual + layer norm

### 4) Decoder layer
Decoder adds two attentions:
1. Masked self-attention (can’t look ahead)
2. Cross-attention over encoder outputs
3. Feed-forward + norms/residuals

### 5) Causal masking in target sequence
The decoder’s self-attention uses an upper-triangular mask so position $t$ cannot attend to future tokens $>t$.

---

## How this notebook builds the Transformer

### A) Setup and utilities
The notebook defines:
- device selection (`cuda` / `mps` / `cpu`),
- training loop (`train`) with LR scheduling (`ReduceLROnPlateau`),
- evaluation utility (`evaluate_tm`).

### B) Dataset
It loads Tatoeba English-Spanish text pairs using Hugging Face `datasets`:
- `ageron/tatoeba_mt_train`
- uses validation split and creates train/valid subsets.

### C) Tokenizer
A BPE tokenizer (`tokenizers`) is trained on both source and target texts with:
- `vocab_size = 10_000`
- special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
- truncation and padding enabled.

### D) Positional embedding
`PositionalEmbedding` is implemented as learned positional vectors added to token embeddings, followed by dropout.

### E) Attention and Transformer blocks
Custom modules implemented in PyTorch:
- `MultiheadAttention`
- `TransformerEncoderLayer`
- `TransformerDecoderLayer`
- `TransformerEncoder` (stack)
- `TransformerDecoder` (stack)
- `Transformer` (full encoder-decoder)

This gives a clean from-scratch architecture implementation.

### F) NMT model wrapper
`NmtTransformer` includes:
- shared token embedding,
- positional embedding,
- sequence-to-sequence transformer core (`nn.Transformer` by default),
- output projection to vocabulary logits.

It also builds:
- source padding mask,
- target padding mask,
- causal mask for decoder self-attention.

### G) Training config
The notebook trains with:
- `CrossEntropyLoss(ignore_index=0)` to ignore `<pad>`
- `NAdam` optimizer
- multiclass accuracy metric (`torchmetrics.Accuracy`)
- configurable epoch count (`n_epochs = 20` in current setup)

It includes an MPS-specific workaround: on Apple Metal devices, it switches to custom `Transformer` implementation due to instability in `nn.Transformer` on some setups.

---

## Environment setup

Recommended:
- Python 3.10+
- PyTorch
- `datasets`
- `tokenizers`
- `torchmetrics`
- `matplotlib`, `numpy`

Install example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch datasets tokenizers torchmetrics matplotlib numpy
```

---

## How to run

1. Open [transformer/Transformer.ipynb](transformer/Transformer.ipynb).
2. Select a Python kernel with the dependencies above.
3. Run cells from top to bottom in order.
4. Verify:
   - tokenizer trains successfully,
   - masks are generated correctly,
   - model trains and logs train/valid metrics each epoch.

---

## Current notebook caveats

1. **`torchmetrics` is used but not imported in visible import cells**
   - Add:
   ```python
   import torchmetrics
   ```

2. **DataLoader/Pair batching objects are referenced in training call**
   - The training cell uses `nmt_train_loader` and `nmt_valid_loader` plus `pair.src_token_ids`/`pair.tgt_token_ids` style fields.
   - Ensure your notebook includes/executes the batch-collation section that creates these objects before training.

3. **Large sequence length may increase memory usage**
   - Current tokenizer truncation length is up to 256.
   - Reduce batch size / embedding dim / layers on constrained hardware.

---

## Why this notebook is valuable

It teaches both:
- **theory**: attention, masking, encoder-decoder flow,
- **implementation**: custom modules + practical training for NMT.

This makes it a strong bridge between conceptual Transformer knowledge and real PyTorch code for translation.

---

## Suggested next improvements

- Add BLEU/chrF evaluation on validation/test translations.
- Add greedy/beam decoding utilities for inference.
- Save best checkpoint by validation metric.
- Add label smoothing for NMT training stability.
- Add mixed precision training for faster GPU runs.

---

If you want, I can also patch [transformer/Transformer.ipynb](transformer/Transformer.ipynb) with a minimal run-safe section that defines/imports any missing training dependencies (`torchmetrics`, dataloader/collate setup order).