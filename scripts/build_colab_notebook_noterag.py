"""Derive notebooks/colab_transformer_training_noterag.ipynb from the original Colab notebook.

The original notebook is read and never modified. The derived notebook trains ONE variant,
notes + note-RAG, using RAG_COLUMN="rag_note_rate_k20" and writes every output file with the
MODEL_TAG="noterag" suffix. The no-RAG transformer already exists and is reused by the grid.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "notebooks" / "colab_transformer_training.ipynb"
TARGET = ROOT / "notebooks" / "colab_transformer_training_noterag.ipynb"

MARKDOWN = """# ReadmitIQ: Transformer with note-embedding RAG feature (Colab GPU)

This notebook fine-tunes **DistilBERT** on clinical discharge notes with one extra input:
the historical readmission rate of the 20 most similar past TRAIN encounters, retrieved over
note embeddings (`rag_note_rate_k20`). It is the "Transformer with RAG" cell of use case 6.

### Methodology
- One variant only: notes + note-RAG. The no-RAG transformer was already trained and is reused.
- Leakage protection: TRAIN rates are out-of-fold (patient-grouped); validation rates query the TRAIN index.
- Evaluation: validation split ONLY. The test split is never touched.
- The last epoch is reported; no cherry-picking across epochs.

### Instructions (Google Colab, T4 GPU)
1. Runtime -> Change runtime type -> T4 GPU.
2. Upload from `data/processed/colab_upload_noterag/`:
   `colab_train_noterag.parquet`, `colab_val_noterag.parquet`, `MANIFEST_noterag.json`.
3. Run all cells.
4. Unzip the downloaded `colab_outputs_noterag.zip` into `reports/note_rag/colab/` in the repo.
5. Run `python scripts/build_six_model_grid.py`.
"""

CONFIG_ADDITION = """
# ---- note-RAG settings (the only configuration that differs from the original notebook) ----
RAG_COLUMN = "rag_note_rate_k20"  # column of the uploaded parquets used as the retrieval feature
MODEL_TAG = "noterag"             # suffix of every uploaded and written file
VARIANT_NAME = f"notes_{MODEL_TAG}"
"""

TRAIN_CELL = """# ==============================================================================
# STEP 5: Train the notes + note-RAG variant
# ==============================================================================
# Subsample train set if TRAIN_N is specified (>= 25,000)
if TRAIN_N is not None:
    assert TRAIN_N >= 25000, f"TRAIN_N must be >= 25000, got {TRAIN_N}"
    pos_train = train_df[train_df["readmitted_30d"] == 1]
    neg_train = train_df[train_df["readmitted_30d"] == 0]
    pos_ratio = len(pos_train) / len(train_df)
    pos_cnt = int(TRAIN_N * pos_ratio)
    neg_cnt = TRAIN_N - pos_cnt
    active_train = pd.concat([
        pos_train.sample(n=pos_cnt, random_state=SEED),
        neg_train.sample(n=neg_cnt, random_state=SEED),
    ]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Training on stratified subset of {len(active_train):,} notes.")
else:
    active_train = train_df.copy()
    print(f"Training on FULL training set of {len(active_train):,} notes.")

# Validation split is NEVER subsampled: exactly 14,902 rows
active_val = val_df.copy()
assert len(active_val) == 14902, f"Validation set must be full 14,902 rows, got {len(active_val)}"
assert RAG_COLUMN in active_train.columns and RAG_COLUMN in active_val.columns, RAG_COLUMN

# Prefix keeps the exact wording of the structured-RAG variant so the two are comparable.
train_rag_texts = [
    f"Similar past patients readmission rate: {r:.0%}. " + note
    for r, note in zip(active_train[RAG_COLUMN], active_train["clinical_note"])
]
val_rag_texts = [
    f"Similar past patients readmission rate: {r:.0%}. " + note
    for r, note in zip(active_val[RAG_COLUMN], active_val["clinical_note"])
]

run_noterag = train_and_eval_variant(
    variant_name=VARIANT_NAME,
    train_texts=train_rag_texts,
    train_labels=active_train["readmitted_30d"].tolist(),
    train_encounters=active_train["encounter_id"].tolist(),
    val_texts=val_rag_texts,
    val_labels=active_val["readmitted_30d"].tolist(),
    val_encounters=active_val["encounter_id"].tolist(),
)

print("=" * 80)
print(f"{VARIANT_NAME} Val AUC (last epoch): {run_noterag['last_epoch_val_auc']:.4f}")
print("=" * 80)
"""

ZIP_CELL_FILES_OLD = """files_to_pack = [
    "distilbert_val_predictions_notes.csv",
    "distilbert_val_predictions_notes_rag.csv",
    "run_notes.json",
    "run_notes_rag.json",
]"""
ZIP_CELL_FILES_NEW = """files_to_pack = [
    f"distilbert_val_predictions_{VARIANT_NAME}.csv",
    f"run_{VARIANT_NAME}.json",
]"""


def source_of(cell: Dict[str, Any]) -> str:
    return "".join(cell["source"])


def set_source(cell: Dict[str, Any], text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new)


def derive(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells = copy.deepcopy(cells)
    cells[0]["source"] = MARKDOWN.splitlines(keepends=True)

    config = source_of(cells[1])
    config = replace_once(config, "SEED = 42\n", "SEED = 42\n" + CONFIG_ADDITION)
    set_source(cells[1], config)

    load = source_of(cells[3])
    load = replace_once(
        load, 'DATA_DIR / "colab_train.parquet"', 'DATA_DIR / f"colab_train_{MODEL_TAG}.parquet"'
    )
    load = replace_once(
        load, 'DATA_DIR / "colab_val.parquet"', 'DATA_DIR / f"colab_val_{MODEL_TAG}.parquet"'
    )
    load = replace_once(
        load, 'DATA_DIR / "MANIFEST.json"', 'DATA_DIR / f"MANIFEST_{MODEL_TAG}.json"'
    )
    load = replace_once(
        load, 'manifest["files"]["colab_train.parquet"]', 'manifest["files"][train_file.name]'
    )
    load = replace_once(
        load, 'manifest["files"]["colab_val.parquet"]', 'manifest["files"][val_file.name]'
    )
    set_source(cells[3], load)

    set_source(cells[5], TRAIN_CELL)

    pack = source_of(cells[6])
    pack = replace_once(pack, ZIP_CELL_FILES_OLD, ZIP_CELL_FILES_NEW)
    pack = pack.replace("colab_outputs.zip", "colab_outputs_noterag.zip")
    pack = pack.replace("reports/colab/", "reports/note_rag/colab/")
    pack = pack.replace("unzip the 4 files", "unzip the 2 files")
    pack = replace_once(
        pack,
        "2. Run: python -m src.transformer_import",
        "2. Run: python scripts/build_six_model_grid.py",
    )
    pack = replace_once(pack, 'print("3. Run: python scripts/sync_transformer_docs.py")\n', "")
    set_source(cells[6], pack)
    return cells


def main() -> int:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook["cells"] = derive(notebook["cells"])
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
