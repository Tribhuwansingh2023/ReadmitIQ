# Note-RAG runbook (embeddings, features, ablation)

Goal: build the note-embedding retrieval feature (`rag_note_rate_k20` and six siblings) and compare
Logistic Regression, Random Forest and the Transformer with and without it, on the validation split.
The frozen production model is never changed. The test split is never evaluated.

Everything below is copy-paste. Replace nothing except paths if your repo is elsewhere.

## 0. One-time setup (any machine with the repo)

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt sqlite-vec     # Windows: .venv/Scripts/python.exe
```

Protect the frozen artifacts: take the hash baseline once, before anything else.

```bash
.venv/bin/python scripts/check_protected_files.py snapshot
```

Write the notes v2 files (CPU, seconds). They need `data/processed/{train,val,test}.parquet`:

```bash
.venv/bin/python -m src.notes_v2
```

This writes `data/processed/note_rag/notes_v2_{train,val,test}.parquet` and fails if any check fails
(label words, template drift, "adult" for minors, row counts).

## 1. Embeddings, path A: Jina API (no GPU)

Model: `JINA_MODEL` (default `jina-embeddings-v5-text-small`). Put the key in the git-ignored `.env`
at the repo root. It is never printed, logged or saved to any output.

```bash
# .env (already ignored by git)
JINA_API_KEY=<your key>
JINA_MODEL=jina-embeddings-v5-text-small
JINA_RERANK_MODEL=jina-reranker-v2-base-multilingual
```

Smoke test first (300 notes, separate output folder). It prints the tokens billed:

```bash
.venv/bin/python scripts/embed_notes_jina.py --variant v2 --limit 300 --out /tmp/jina_smoke
```

Estimate the full cost from the smoke run: `billed_tokens / 300 * total_rows`, with total rows from
`data/processed/feature_metadata.json`. Identical notes are sent only once (SQLite cache), so the real
cost is at or below the estimate. Then run for real, with a budget guard you choose:

```bash
nohup .venv/bin/python scripts/embed_notes_jina.py --variant v2 --token-budget 20000000 > embed_jina_v2.log 2>&1 &
tail -f embed_jina_v2.log
# optional second retriever on the v1 notes
nohup .venv/bin/python scripts/embed_notes_jina.py --variant v1 --token-budget 30000000 > embed_jina_v1.log 2>&1 &
```

If the budget stops the run, re-run the same command: cached vectors are kept and only the rest is billed.

Outputs in `data/processed/note_rag/`: `embeddings_jina_v5_small_{variant}_{split}.npy`,
`embeddings_jina_v5_small_{variant}_index.parquet`, `MANIFEST.json`, and the cache
`embedding_cache_jina_v5_small.sqlite`.

## 2. Embeddings, path B: GPU machine (VPS or Colab) with local models

Upload exactly these files, keeping this layout:

```
data/processed/train.parquet
data/processed/val.parquet
data/processed/test.parquet
data/processed/note_rag/notes_v2_train.parquet
data/processed/note_rag/notes_v2_val.parquet
data/processed/note_rag/notes_v2_test.parquet
scripts/embed_notes.py
requirements-note-rag.txt
```

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements-note-rag.txt
nohup .venv/bin/python scripts/embed_notes.py --model minilm    --variant v2 --out out > minilm_v2.log 2>&1 &
tail -f minilm_v2.log
```

Repeat with `--variant v1`, and with `--model bge_small` for both variants (run them one after another;
all four share `out/MANIFEST.json`). `--batch-size auto` is the default; `--limit 300` gives a smoke test.
The script warns loudly on CPU and resumes by skipping any split whose `.npy` matches the manifest checksum.

Disk per embeddings file: `rows x dim x 4` bytes (the dimension is printed in the manifest). Runtime is
printed per split as rows per second.

Copy back into `data/processed/note_rag/` on your machine: every `embeddings_*.npy`, every
`embeddings_*_index.parquet`, and `MANIFEST.json`.

## 3. After the embeddings are in place (local, CPU)

```bash
.venv/bin/python scripts/select_note_retriever.py        # TRAIN out-of-fold only; writes reports/note_rag/DECISION.md
.venv/bin/python scripts/build_note_rag_features.py      # features, assertions, SQLite RAG store
cat reports/note_rag/note_rag_assertions.json            # every check must pass
.venv/bin/python scripts/run_note_rag_ablation.py        # LR, RF, LightGBM on the validation split
.venv/bin/python scripts/build_six_model_grid.py         # six-model grid; missing artifacts show as pending
.venv/bin/python scripts/check_protected_files.py verify # protected files unchanged
```

The SQLite RAG store is `models/note_rag/note_rag_train.sqlite` (TRAIN rows only, cosine, with patient and
label metadata). `run_note_rag_ablation.py` stops if the "without RAG" rows do not reproduce
`ablation_table.json` within 0.0005.

## 4. Transformer with the note-RAG feature (Colab T4)

```bash
.venv/bin/python scripts/prepare_colab_upload_noterag.py
```

Upload `data/processed/colab_upload_noterag/{colab_train_noterag.parquet, colab_val_noterag.parquet, MANIFEST_noterag.json}`
and open `notebooks/colab_transformer_training_noterag.ipynb`. The configuration cell holds
`RAG_COLUMN = "rag_note_rate_k20"` and `MODEL_TAG = "noterag"`. Run all cells, unzip the downloaded
`colab_outputs_noterag.zip` into `reports/note_rag/colab/`, then re-run `scripts/build_six_model_grid.py`.
The notebook is regenerated from the original with `scripts/build_colab_notebook_noterag.py`.

## 5. Known environment issues

- Windows: `import torch` can fail with WinError 1114 depending on import order. It works when
  scikit-learn is imported first. Linux and Colab are not affected.
- Windows: LightGBM 4.7.0 crashed with an access violation when building a dataset; 4.6.0 works.
  Install `lightgbm==4.6.0` there (the repo lockfile is not edited).
