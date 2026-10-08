"""Background note embedding script using all-MiniLM-L6-v2 on CPU.

Processes train (69,535), val (14,902), test (14,903) sequentially.
Saves embeddings_*.npy in exact parquet row order with checksum metadata.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("STARTING FULL-SCALE CLINICAL NOTE EMBEDDING (all-MiniLM-L6-v2, CPU)")
    print("=" * 70)

    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer '{model_name}' on CPU...")
    model = SentenceTransformer(model_name, device="cpu")

    splits = ["train", "val", "test"]
    checksums = {}

    for split in splits:
        parquet_path = PROCESSED_DATA_DIR / f"{split}.parquet"
        out_npy = PROCESSED_DATA_DIR / f"embeddings_{split}.npy"

        df = pd.read_parquet(parquet_path)
        notes = df["clinical_note"].tolist()
        n_notes = len(notes)
        print(f"\n[{split.upper()}] Encoding {n_notes} notes...")

        t0 = time.time()
        embs = model.encode(
            notes,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        elapsed = time.time() - t0

        assert embs.shape == (n_notes, 384), f"Shape mismatch: {embs.shape} vs ({n_notes}, 384)"
        np.save(out_npy, embs)
        print(f"[{split.upper()}] Saved {out_npy.name} (shape: {embs.shape}) in {elapsed/60:.1f} mins.")

        checksums[split] = {
            "rows": n_notes,
            "embedding_dim": 384,
            "file": out_npy.name,
            "sha256": file_sha256(out_npy),
            "encoding_time_seconds": round(elapsed, 2),
        }

    checksum_path = PROCESSED_DATA_DIR / "embeddings_checksum.json"
    with open(checksum_path, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"\nCompleted all splits! Checksum saved to {checksum_path}")


if __name__ == "__main__":
    main()
