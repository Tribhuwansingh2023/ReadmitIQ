"""Prepare small, focused dataset parquets for Google Colab transformer training.

Extracts only the columns needed for fine-tuning DistilBERT (encounter_id,
clinical_note, readmitted_30d, rag_struct_rate_k20) from the processed
train_rag and val_rag parquets. Enforces strict integrity assertions:
- Exact row counts (69,535 train, 14,902 val)
- No missing / null values
- Zero target label word leakage in clinical notes
- SHA-256 manifest generation for verification in Colab
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, TRAIN_RAG_PARQUET, VAL_RAG_PARQUET

COLAB_UPLOAD_DIR = PROCESSED_DATA_DIR / "colab_upload"
COLAB_TRAIN_PARQUET = COLAB_UPLOAD_DIR / "colab_train.parquet"
COLAB_VAL_PARQUET = COLAB_UPLOAD_DIR / "colab_val.parquet"
MANIFEST_PATH = COLAB_UPLOAD_DIR / "MANIFEST.json"

REQUIRED_COLUMNS: List[str] = [
    "encounter_id",
    "clinical_note",
    "readmitted_30d",
    "rag_struct_rate_k20",
]

EXPECTED_TRAIN_ROWS: int = 69_535
EXPECTED_VAL_ROWS: int = 14_902

FORBIDDEN_LABEL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\breadmit\b", re.IGNORECASE),
    re.compile(r"\breadmission\b", re.IGNORECASE),
    re.compile(r"\breadmitted\b", re.IGNORECASE),
    re.compile(r"(?<!\d)<30(?!\d)"),
    re.compile(r"(?<!\d)>30(?!\d)"),
    re.compile(r"\boutcome\b", re.IGNORECASE),
    re.compile(r"\brecurrent\b", re.IGNORECASE),
]


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_no_label_leakage(notes: pd.Series, split_name: str) -> None:
    """Verify that clinical discharge notes contain zero label words."""
    for pattern in FORBIDDEN_LABEL_PATTERNS:
        matches = notes.str.contains(pattern, regex=True)
        num_matches = int(matches.sum())
        assert (
            num_matches == 0
        ), f"Label leakage detected in {split_name}: {num_matches} notes matched pattern {pattern.pattern}"


def extract_and_validate_split(
    source_parquet: Path,
    expected_rows: int,
    split_name: str,
) -> pd.DataFrame:
    """Load split parquet, extract required columns, and validate schema and contents."""
    assert source_parquet.exists(), f"Source parquet not found: {source_parquet}"
    df = pd.read_parquet(source_parquet)

    # Verify encounter_id exists; if not present, check cleaned data
    if "encounter_id" not in df.columns:
        cleaned_parquet = PROCESSED_DATA_DIR / f"{split_name}.parquet"
        assert cleaned_parquet.exists(), f"Fallback cleaned file not found: {cleaned_parquet}"
        cleaned_df = pd.read_parquet(cleaned_parquet, columns=["encounter_id"])
        assert len(cleaned_df) == len(
            df
        ), f"Row count mismatch between {split_name} and cleaned data"
        df["encounter_id"] = cleaned_df["encounter_id"].values

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Required column '{col}' missing from {source_parquet}"

    selected = df[REQUIRED_COLUMNS].copy()

    # Integrity assertions
    assert (
        len(selected) == expected_rows
    ), f"Expected {expected_rows} rows for {split_name}, got {len(selected)}"
    null_counts = selected.isnull().sum()
    assert null_counts.sum() == 0, f"Null values detected in {split_name}: {null_counts.to_dict()}"

    # Verify notes contain no label words
    verify_no_label_leakage(selected["clinical_note"], split_name)

    return selected


def create_colab_upload_package(
    train_source: Path = TRAIN_RAG_PARQUET,
    val_source: Path = VAL_RAG_PARQUET,
    output_dir: Path = COLAB_UPLOAD_DIR,
) -> Dict[str, Any]:
    """Build colab_train.parquet, colab_val.parquet and MANIFEST.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and extracting training data from {train_source}...")
    train_df = extract_and_validate_split(train_source, EXPECTED_TRAIN_ROWS, "train")

    print(f"Loading and extracting validation data from {val_source}...")
    val_df = extract_and_validate_split(val_source, EXPECTED_VAL_ROWS, "val")

    train_out = output_dir / "colab_train.parquet"
    val_out = output_dir / "colab_val.parquet"
    manifest_out = output_dir / "MANIFEST.json"

    print(f"Saving {train_out} ({len(train_df):,} rows)...")
    train_df.to_parquet(train_out, index=False, engine="pyarrow")

    print(f"Saving {val_out} ({len(val_df):,} rows)...")
    val_df.to_parquet(val_out, index=False, engine="pyarrow")

    train_sha = compute_sha256(train_out)
    val_sha = compute_sha256(val_out)
    train_bytes = train_out.stat().st_size
    val_bytes = val_out.stat().st_size

    manifest_data: Dict[str, Any] = {
        "dataset_name": "ReadmitIQ Colab Transformer Upload",
        "columns": REQUIRED_COLUMNS,
        "files": {
            "colab_train.parquet": {
                "rows": len(train_df),
                "columns": REQUIRED_COLUMNS,
                "size_bytes": train_bytes,
                "size_mb": round(train_bytes / (1024 * 1024), 2),
                "sha256": train_sha,
            },
            "colab_val.parquet": {
                "rows": len(val_df),
                "columns": REQUIRED_COLUMNS,
                "size_bytes": val_bytes,
                "size_mb": round(val_bytes / (1024 * 1024), 2),
                "sha256": val_sha,
            },
        },
    }

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Manifest written to {manifest_out}")
    print(f"Train SHA-256: {train_sha} ({train_bytes / (1024 * 1024):.2f} MB)")
    print(f"Val SHA-256:   {val_sha} ({val_bytes / (1024 * 1024):.2f} MB)")
    return manifest_data


if __name__ == "__main__":
    create_colab_upload_package()
