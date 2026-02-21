# Transformer Encoder for IMDB Sentiment Classification

This folder contains [TransformerEncoder.ipynb](TransformerEncoder.ipynb), a practical notebook that explains and implements a Transformer Encoder for binary sentiment classification.

The notebook progresses from:
1. text dataset loading,
2. integer tokenization/vectorization,
3. a custom Transformer encoder block,
4. a classifier head,
5. an extended version with positional embedding.

---

## Project structure

```text
transformer-encoder-model/
├── TransformerEncoder.ipynb
└── README.md
```

Artifacts produced when running the notebook:
- `transformer_encoder.keras`
- `full_transformer_encoder.keras`

---

## What is a Transformer Encoder?

A Transformer Encoder is a sequence-processing block built from:

- multi-head self-attention,
- a feed-forward network,
- residual (skip) connections,
- layer normalization.

For each token position, self-attention allows the model to look at all other token positions and compute context-aware representations.

### Core encoder equations (high level)

Given input sequence representation $X$:

1. Self-attention + residual + norm
$$
H_1 = \text{LayerNorm}(X + \text{MHA}(X, X, X))
$$

2. Feed-forward + residual + norm
$$
H_2 = \text{LayerNorm}(H_1 + \text{FFN}(H_1))
$$

This pattern is exactly what your custom `TransformerEncoder` layer implements.

---

## How this notebook builds Transformer Encoder

### 1) Data loading
The notebook loads IMDB sentiment data from local folders using:
- `keras.utils.text_dataset_from_directory`
- `aclImdb/train`, `aclImdb/val`, and `aclImdb/test`

### 2) Text vectorization
It uses `TextVectorization` with:
- `max_tokens = 20000`
- `output_mode = "int"`
- `output_sequence_length = 600`

This converts raw text into fixed-length integer token sequences.

### 3) Custom `TransformerEncoder` layer
The notebook defines a custom Keras layer with:
- `MultiHeadAttention(num_heads, key_dim=embed_dim)`
- feed-forward projection (`Dense(dense_dim, relu)` then `Dense(embed_dim)`)
- two `LayerNormalization` operations
- optional mask handling for padded tokens
- `get_config()` for serialization support

### 4) First model (embedding + encoder + classifier)
Pipeline:
- token IDs input
- token embedding
- custom Transformer encoder block
- global max pooling
- dropout
- sigmoid output for binary classification

Compiled with:
- optimizer: `rmsprop`
- loss: `binary_crossentropy`
- metric: `accuracy`

### 5) Positional embedding upgrade
Notebook adds a `PositionalEmbedding` custom layer that:
- learns token embeddings,
- learns positional embeddings,
- sums both to inject order information,
- propagates a padding mask via `compute_mask`.

Then it attempts a full encoder model with positional information.

---

## Why positional embedding matters

Self-attention alone is permutation-invariant with respect to positions. Without positional information, token order is ambiguous.

The `PositionalEmbedding` layer restores order-awareness by adding a learned vector for each position index.

---

## Environment setup

Recommended:
- Python 3.10+
- TensorFlow 2.13+ (or compatible 2.x)
- Jupyter or VS Code notebook kernel

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow
```

---

## Data requirements

This notebook expects the IMDB directory structure to already exist:

```text
aclImdb/
  train/
  val/
  test/
```

If `aclImdb` is not present, create/populate it first (for example, using your earlier sequence-model notebook workflow).

---

## How to run

1. Open [TransformerEncoder.ipynb](TransformerEncoder.ipynb).
2. Select a Python kernel with TensorFlow installed.
3. Ensure `aclImdb` folders are available.
4. Run cells from top to bottom.
5. Verify:
   - model summary appears,
   - training runs,
   - best checkpoints are saved,
   - test accuracy prints.

---

## Known caveats in current notebook

1. **Second model currently uses `Lambda` incorrectly**
   - In the positional-embedding section, `Lambda` returns layer objects instead of applying layers to tensors.
   - This is why that cell can error.
   - Correct form should directly call layers:

```python
embedded = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(embedded)
```

2. **No explicit download/split code for IMDB in this notebook**
   - It assumes `aclImdb/val` already exists.

3. **Potential duplicate imports**
   - TensorFlow/Keras imports appear multiple times; harmless but can be cleaned.

---

## Expected outcomes

After successful runs, you should observe:
- a baseline Transformer encoder sentiment model checkpoint,
- an improved position-aware variant once the `Lambda` issue is corrected,
- printed test accuracy values for each run.

---

## Suggested next improvements

- Replace `GlobalMaxPooling1D` with attention pooling or CLS-token style pooling and compare.
- Add `EarlyStopping` callback to reduce overfitting.
- Add precision/recall/F1 in evaluation.
- Stack multiple encoder blocks to increase capacity.
- Compare against BiLSTM baseline on identical preprocessing.

---

If you want, I can patch [TransformerEncoder.ipynb](TransformerEncoder.ipynb) now with a minimal fix for the positional-embedding model so both training pipelines run end to end.