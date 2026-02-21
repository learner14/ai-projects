# Sequence Model for IMDB Sentiment Classification

This folder contains `Implementing_SequenceModel.ipynb`, a practical notebook that builds and compares multiple sequence-modeling approaches for binary sentiment classification on the IMDB dataset.

It covers:
- one-hot encoded sequence input + BiLSTM,
- trainable embedding layer + BiLSTM,
- embedding masking with `mask_zero=True`,
- pretrained GloVe embeddings + BiLSTM.

---

## Project structure

```text
Sequence-model/
├── Implementing_SequenceModel.ipynb
└── README.md
```

Generated artifacts after running the notebook:
- `aclImdb/` (dataset extracted from tarball)
- `one_hot_bidir_lstm.keras`
- `embeddings_bidir_gru.keras` (filename says GRU, model is BiLSTM)
- `embeddings_bidir_gru_with_masking.keras` (filename says GRU, model is BiLSTM)
- `glove_embeddings_sequence_model.keras`

---

## What this notebook demonstrates

1. **Data acquisition & split prep**
   - Downloads IMDB reviews dataset.
   - Removes unsupervised split (`aclImdb/train/unsup`).
   - Creates validation split by moving 20% of train files into `aclImdb/val`.

2. **Text pipeline**
   - Uses `TextVectorization` with:
     - `max_tokens = 20000`
     - `output_mode = "int"`
     - `output_sequence_length = 600`

3. **Modeling experiments**
   - **Experiment A:** One-hot input representation + BiLSTM.
   - **Experiment B:** Learned embeddings from scratch + BiLSTM.
   - **Experiment C:** Learned embeddings with masking + BiLSTM.
   - **Experiment D:** Frozen pretrained GloVe embedding matrix + BiLSTM.

4. **Training/evaluation**
   - Binary classification objective with `binary_crossentropy`.
   - Optimizer: `rmsprop`.
   - Tracks validation performance and restores best checkpoint.

---

## Environment setup

### Recommended
- Python 3.10+
- Jupyter / VS Code Notebook
- TensorFlow 2.13+ (or compatible Keras/TensorFlow combination)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy
```

Optional utility packages:
```bash
pip install jupyter matplotlib
```

---

## External data requirements

### 1) IMDB sentiment dataset
Notebook fetches:
- `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`

### 2) GloVe vectors
Notebook expects (for pretrained embedding section):
- `data/glove.6B.100d.txt`

If missing, download and unzip GloVe so that file path matches exactly:
```text
data/glove.6B.100d.txt
```

---

## How to run

1. Open `Implementing_SequenceModel.ipynb`.
2. Select Python kernel with TensorFlow installed.
3. Run cells top-to-bottom once.
4. Confirm checkpoints are created and test accuracy is printed after each experiment.

---

## Typical workflow inside the notebook

- Build `train_ds`, `val_ds`, `test_ds` via `keras.utils.text_dataset_from_directory`.
- Adapt `TextVectorization` on train text only.
- Map datasets to integer token sequences.
- Train/evaluate:
  - one-hot model,
  - embedding model,
  - masking model,
  - GloVe-initialized frozen embedding model.

---

## Known caveats and fixes

1. **Validation directory creation error on rerun**
   - The split cell uses `os.makedirs(val_dir / category)` without `exist_ok=True`.
   - Rerunning may raise `FileExistsError`.
   - Fix: use `os.makedirs(val_dir / category, exist_ok=True)`.

2. **Dataset split cell is not idempotent**
   - Files are moved from train → val each run.
   - Re-running without resetting `aclImdb/` can fail or create inconsistent splits.
   - Recommended: delete and re-extract dataset before rerun, or guard split logic.

3. **GloVe path must exist**
   - The notebook uses `path_to_glove_file = "data/glove.6B.100d.txt"`.
   - If file is elsewhere, update this path.

4. **Checkpoint filenames mention GRU**
   - Some filenames include `gru`, but architecture used is BiLSTM.
   - This is naming only; training logic still works.

5. **Embedding matrix construction indentation risk**
   - In the `embedding_matrix` loop, `embedding_vector` check should be nested with the `i < max_tokens` branch to avoid stale values:

```python
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
```

---

## Expected outputs

After successful execution, you should see:
- sample raw review + label,
- model summaries for each experiment,
- training logs per epoch,
- test accuracy values for each model variant,
- saved `.keras` checkpoint files.

---

## Suggested next improvements

- Add precision/recall/F1 alongside accuracy.
- Use `tf.data.AUTOTUNE` and prefetch/cache for faster input pipelines.
- Add early stopping callback.
- Compare BiLSTM vs BiGRU with identical settings.
- Unfreeze GloVe embeddings for a final fine-tuning stage.

---

If you want, I can also patch the notebook to make it rerun-safe (idempotent split + safer GloVe embedding loop) in a minimal way.