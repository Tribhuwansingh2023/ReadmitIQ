"""Prepare the Colab upload for the transformer with the note-RAG feature.

Builds colab_train_noterag.parquet and colab_val_noterag.parquet with
encounter_id, clinical_note (v1 notes, as the original transformer used), readmitted_30d and
rag_note_rate_k20, plus MANIFEST_noterag.json. Run it after build_note_rag_features.py.
The existing prepare_colab_upload.py and its outputs are not touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_colab_upload import compute_sha256, verify_no_label_leakage  # noqa: E402

from src.note_rag_config import (  # noqa: E402
    ID_COL,
    LABEL_COL,
    NOTE_RAG_REPORT_DIR,
    PRIMARY_FEATURE,
    PROCESSED_DIR,
    expected_split_rows,
    features_path,
    split_parquet,
)

OUT_DIR = PROCESSED_DIR / "colab_upload_noterag"
TAG = "noterag"
COLUMNS = [ID_COL, "clinical_note", LABEL_COL, PRIMARY_FEATURE]


def build_split(split: str, expected_rows: int) -> pd.DataFrame:
    base = pd.read_parquet(split_parquet(split), columns=[ID_COL, "clinical_note", LABEL_COL])
    feats = pd.read_parquet(features_path(split), columns=[ID_COL, PRIMARY_FEATURE])
    if len(base) != expected_rows or len(feats) != expected_rows:
        raise AssertionError(
            f"{split}: expected {expected_rows} rows, got {len(base)}/{len(feats)}"
        )
    if not base[ID_COL].equals(feats[ID_COL]):
        raise AssertionError(f"{split}: feature encounter order differs from the split parquet")
    out = pd.concat([base, feats[[PRIMARY_FEATURE]]], axis=1)[COLUMNS]
    if out.isnull().any().any():
        raise AssertionError(f"{split}: null values in the upload columns")
    verify_no_label_leakage(out["clinical_note"], split)
    return out


def main() -> int:
    rows = expected_split_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = NOTE_RAG_REPORT_DIR / "selected_retriever.json"
    retriever = (
        json.loads(selection.read_text(encoding="utf-8"))["selected"]
        if selection.exists()
        else None
    )
    manifest: Dict[str, Any] = {
        "dataset_name": "ReadmitIQ Colab note-RAG upload",
        "retriever": retriever,
        "files": {},
    }
    for split in ("train", "val"):
        df = build_split(split, rows[split])
        path = OUT_DIR / f"colab_{split}_{TAG}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        manifest["files"][path.name] = {
            "rows": len(df),
            "columns": COLUMNS,
            "size_bytes": path.stat().st_size,
            "sha256": compute_sha256(path),
        }
        print(f"{path.name}: {len(df)} rows")
    (OUT_DIR / f"MANIFEST_{TAG}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {OUT_DIR.name}/MANIFEST_{TAG}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
