# Google Colab Transformer Fine-Tuning Runbook

A complete, numbered, copy-paste execution checklist for running the ReadmitIQ clinical notes transformer fine-tuning workflow on a Google Colab T4 GPU and importing the results locally.

---

## Overview & Scientific Expectation

- **Objective:** Fine-tune `distilbert-base-uncased` on clinical discharge notes for 30-day hospital readmission risk prediction, comparing **Variant A (Notes Only)** against **Variant B (Notes + RAG structured prevalence)**.
- **Evaluation Split:** Evaluated strictly on the full **14,902 validation split** (`colab_val.parquet`) with paired bootstrap confidence intervals ($B=1,000$).
- **Methodological Integrity:** **The production model was frozen before the transformer results existed.** This transformer benchmark is an offline validation-only comparison; the held-out test split is never evaluated.
- **Honest Expectation:** The clinical discharge notes reflect the same patient acuity and prior utilization already captured in the 41 tabular features. Expect modest AUC-ROC values in the tabular range (~0.64–0.68). This is normal in clinical EHR modeling and must be reported exactly as measured.

---

## Step-by-Step Execution Checklist

### 1. Open Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **File** $\rightarrow$ **Upload notebook**.
3. Select `notebooks/colab_transformer_training.ipynb` from this repository.

### 2. Enable Free T4 GPU Accelerator
1. In Colab, click **Runtime** in the top menu $\rightarrow$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

### 3. Upload Required Files to Colab
In the left sidebar of Colab, click the **Files (folder icon)** tab. Upload the following **3 small files** directly from `data/processed/colab_upload/`:

| File Name | Size | Sample Count | SHA-256 Checksum |
|:---|:---:|:---:|:---|
| `colab_train.parquet` | **5.64 MB** | 69,535 rows | `040a41e2a086b182c600d0bca83f9400b7a938a7102dacfc241e8940c24182c1` |
| `colab_val.parquet` | **1.22 MB** | 14,902 rows | `0438ebb30233a62f04719e90460c381a9d498fb638eb4ab4ea123768f2be5213` |
| `MANIFEST.json` | **866 B** | N/A | Metadata & verification hashes |

*(Total upload size is only ~6.86 MB and completes in under 5 seconds).*

> [!NOTE]
> If you upload them to `/content/` (the default Colab root), the notebook will find them immediately (`DATA_DIR = Path(".")`). If you mount Google Drive, you can optionally store them in Google Drive.

### 4. Review Hyperparameters (Cell 1)
The top configuration cell has sensible defaults pre-configured:
```python
MODEL_NAME = "distilbert-base-uncased"
TRAIN_N = None          # None = Full 69,535 notes; or set to 25000 for faster training
EPOCHS = 3              # 3 full epochs per variant
BATCH_SIZE = 32         # Optimized for T4 15 GB VRAM
LR = 3e-5               # AdamW learning rate
MAX_LENGTH = 192        # Covers 100% of notes (0.0% truncated)
SEED = 42
DATA_DIR = Path(".")
OUT_DIR = Path("./outputs")
```

### 5. Run All Cells Top to Bottom
Click **Runtime** $\rightarrow$ **Run all** (or press `Ctrl+F9`).

#### Expected Execution Times:
- **Environment & Hugging Face download:** ~1 minute.
- **Variant A (Notes Only, 3 epochs):** ~10–12 minutes.
- **Variant B (Notes + RAG, 3 epochs):** ~10–12 minutes.
- **Total Runtime:** **~22–26 minutes** on a free T4 GPU.

#### Expected Output Logs:
- `[INFO] GPU detected: Tesla T4 (VRAM: 15.0 GB)`
- `[INFO] Verified colab_train.parquet (SHA-256 match, rows: 69535)`
- `[INFO] Verified colab_val.parquet (SHA-256 match, rows: 14902)`
- `[INFO] Tokenizer max token length: 189. Truncation share at MAX_LENGTH=192: 0.00%`
- `Variant A - Epoch 1/3: Train Loss = 0.58xx, Val AUC = 0.65xx`
- `Variant A - Epoch 2/3: Train Loss = 0.54xx, Val AUC = 0.66xx`
- `Variant A - Epoch 3/3: Train Loss = 0.49xx, Val AUC = 0.66xx`
- `[INFO] Saved distilbert_val_predictions_notes.csv and run_notes.json`
- `Variant B - Epoch 1/3: Train Loss = 0.57xx, Val AUC = 0.65xx`
- `Variant B - Epoch 2/3: Train Loss = 0.53xx, Val AUC = 0.66xx`
- `Variant B - Epoch 3/3: Train Loss = 0.48xx, Val AUC = 0.66xx`
- `[INFO] Saved distilbert_val_predictions_notes_rag.csv and run_notes_rag.json`
- `[SUCCESS] Created colab_outputs.zip`

### 6. Download Output Bundle
Cell 7 automatically triggers the download of `colab_outputs.zip`.
If the automatic browser download is blocked by a popup blocker:
1. In the Colab file tree on the left, locate `colab_outputs.zip`.
2. Right click $\rightarrow$ **Download**.

### 7. Unzip Outputs Locally
Extract the contents of `colab_outputs.zip` directly into the repository folder:
```
reports/colab/
```
Verify that `reports/colab/` contains exactly these 4 files:
- `reports/colab/distilbert_val_predictions_notes.csv` (14,902 rows)
- `reports/colab/distilbert_val_predictions_notes_rag.csv` (14,902 rows)
- `reports/colab/run_notes.json`
- `reports/colab/run_notes_rag.json`

### 8. Run Local Import & Doc Synchronization
Open your local terminal in the `readmitt-iq` workspace and execute:

```bash
# 1. Validate CSVs, compute bootstrap CIs and paired deltas, and update ablation table:
python -m src.transformer_import

# 2. Sync honest numbers into README, Model Card, Deck content, and Q&A prep:
python scripts/sync_transformer_docs.py

# 3. Rebuild presentation deck and verify consistency:
python scripts/build_presentation_pack.py
python scripts/check_deck_numbers.py

# 4. Verify tests remain 100% green:
python -m pytest tests/test_transformer_import.py
```

---

## Troubleshooting & Quota Exhaustion Protocols

### Issue A: Google Colab T4 GPU Quota Runs Out
If Colab says *"Cannot connect to GPU backend"* or terminates early:
- **Option 1 (Kaggle GPU - Recommended):**
  1. Open [Kaggle](https://www.kaggle.com/code) $\rightarrow$ **New Notebook**.
  2. In the notebook settings on the right, select **Accelerator: GPU T4 x2**.
  3. Upload `notebooks/colab_transformer_training.ipynb` and the 3 upload files.
  4. Run all cells (Kaggle provides 30 hours/week of free T4 compute).
- **Option 2 (Reduce `TRAIN_N` in Colab):**
  In Cell 1, change:
  ```python
  TRAIN_N = 25000  # Subsample training notes while keeping validation full N=14,902
  ```
  This reduces GPU training time to ~8 minutes total while preserving full statistical validity.

### Issue B: Loss Does Not Decrease in Epoch 2
- The notebook will print a warning:
  `[WARNING] Epoch 2 loss did not decrease relative to Epoch 1.`
- The run will continue to completion and export the last epoch predictions.
- The local import script will still accept and evaluate the predictions honestly without cherry-picking best epochs.

### Issue C: Disconnect During Training
- The notebook immediately saves `distilbert_val_predictions_notes.csv` and `run_notes.json` at the end of Variant A before starting Variant B.
- If a disconnect occurs during Variant B, Variant A results are already saved in `/content/outputs/` (or in Google Drive if mounted).
