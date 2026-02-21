# Word Embedding + RNN Classification (IMDB Sentiment)

This folder contains `Implementing_Word_embedding_RNN_classification.ipynb`, a notebook focused on building a sentiment classifier with:

- pre-trained **GloVe** word vectors,
- sequence padding,
- stacked **LSTM** layers in Keras/TensorFlow.

The notebook currently mixes a complete preprocessing pipeline with a partially implemented model-definition section (see caveats below).

---

## Project structure

```text
word-embedding-RNN/
├── Implementing_Word_embedding_RNN_classification.ipynb
└── README.md
```

Expected runtime data files:
- `data/glove.6B.50d.txt`
- downloaded IMDB dataset cache (via `tensorflow.keras.datasets.imdb`)

---

## Notebook goals

1. Load GloVe embeddings from text file into dictionaries.
2. Load IMDB sentiment dataset (`num_words=10000`).
3. Pad/truncate reviews to fixed length (`maxlen=256`).
4. Build an RNN classifier with pretrained embedding initialization.

---

## Environment setup

### Recommended
- Python 3.10+
- TensorFlow 2.13+ (or a compatible recent version)
- Jupyter Notebook / VS Code notebook kernel

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy
```

---

## Data requirements

### 1) GloVe embeddings file
The notebook expects this exact file path:

```text
data/glove.6B.50d.txt
```

If you have `glove.6B.zip`, extract it and place `glove.6B.50d.txt` in a local `data/` directory.

### 2) IMDB dataset
Loaded automatically from:

- `tensorflow.keras.datasets.imdb.load_data(num_words=10000)`

Internet access is required on first run.

---

## What each section does

### A) Load packages
Imports NumPy, TensorFlow, and Keras IMDB dataset utilities.

### B) Parse GloVe vectors
`read_glove_vecs(glove_file)` builds:
- `words_to_index`
- `index_to_words`
- `word_to_vec_map`

### C) Load and inspect IMDB data
- Loads train/test splits.
- Prints sample counts.
- Uses integer-encoded reviews from Keras tokenizer index.

### D) Pad sequences
Uses post-padding with zeros:

- `maxlen = 256`
- consistent shape for model input.

### E) Model definition skeleton
`createModel(...)` defines a stacked LSTM architecture with dropout and final dense output.

---

## Model architecture (as written)

Inside `createModel`:
- Input: token index sequence (`int32`)
- Embedding: expected to be pretrained GloVe embedding layer
- LSTM(128, `return_sequences=True`)
- Dropout(0.5)
- LSTM(128, `return_sequences=False`)
- Dropout(0.5)
- Dense(5)
- Activation(`sigmoid`)

---

## Important caveats in current notebook

The notebook is **not fully executable end-to-end yet** due to missing pieces:

1. **Missing symbol imports for model section**
   - `Input`, `LSTM`, `Dropout`, `Dense`, `Activation`, `Model` are used but not imported.

2. **Missing function `pretrained_embedding_layer(...)`**
   - `createModel()` calls this function, but it is not defined in the notebook.

3. **Output layer likely mismatched for IMDB binary task**
   - Current final layers are `Dense(5)` + `sigmoid`.
   - IMDB sentiment is binary; typical setup is `Dense(1, activation="sigmoid")`.

4. **No compile/train/evaluate cells currently present**
   - Notebook defines model function but does not show full training loop or metrics output.

Because of the above, this notebook currently behaves as a **partial implementation draft** rather than a finished training script.

---

## How to run current notebook safely

1. Ensure `data/glove.6B.50d.txt` exists.
2. Run preprocessing cells (GloVe parsing, IMDB loading, sequence padding).
3. Treat model-definition cell as draft unless you add missing imports/functions.

---

## Recommended completion steps

To make this notebook fully runnable:

1. Add missing Keras layer/model imports.
2. Implement `pretrained_embedding_layer(word_to_vec_map, word_to_index)`.
3. Align output head for binary classification (`Dense(1, sigmoid)`).
4. Compile model with:
   - `loss="binary_crossentropy"`
   - metrics including `accuracy`
5. Add `model.fit(...)` and `model.evaluate(...)` cells.

---

## Suggested next improvements

- Add train/validation split from training data.
- Add callbacks (`ModelCheckpoint`, `EarlyStopping`).
- Compare frozen vs trainable embedding layer.
- Report precision/recall/F1 in addition to accuracy.

---

If you want, I can patch the notebook in a minimal way to make it fully runnable end-to-end (imports, embedding layer function, binary output head, compile/train/evaluate cells).