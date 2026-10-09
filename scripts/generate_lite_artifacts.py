#!/usr/bin/env python3
"""Generate compact LITE serving artifacts for low-memory free-tier deployments.

Produces:
- models/final/retriever_faiss_lite.index: 5,000 stratified training prototypes (1.28 MB)
- models/final/retriever_meta_lite.parquet: compact metadata for 5,000 prototypes (<70 KB)
- reports/demo/cohort_scored_lite.parquet: top 3,000 scored test encounters for light replay
"""

import io
import os
import sys
from pathlib import Path

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import faiss
import numpy as np
import pandas as pd


def generate_lite_artifacts():
    print("=" * 65)
    print("[*] Generating ReadmitIQ LITE Serving Artifacts (< 2 MB total)")
    print("=" * 65)

    final_dir = ROOT_DIR / "models" / "final"
    meta_path = final_dir / "retriever_meta.parquet"
    faiss_path = final_dir / "retriever_faiss.index"

    if not meta_path.exists() or not faiss_path.exists():
        print("Error: full retriever artifacts not found.")
        sys.exit(1)

    # 1. Load full metadata and FAISS index
    df_meta = pd.read_parquet(meta_path)
    full_index = faiss.read_index(str(faiss_path))
    n_full = full_index.ntotal
    print(f"Loaded full FAISS index: {n_full:,} vectors, metadata: {len(df_meta):,} rows")

    # 2. Select 5,000 stratified prototypes preserving readmission ratio
    n_lite = 5000
    pos_meta = df_meta[df_meta["readmitted_30d"] == 1]
    neg_meta = df_meta[df_meta["readmitted_30d"] == 0]

    pos_ratio = len(pos_meta) / len(df_meta)
    n_pos = int(n_lite * pos_ratio)
    n_neg = n_lite - n_pos

    sample_pos = pos_meta.sample(n=min(n_pos, len(pos_meta)), random_state=42)
    sample_neg = neg_meta.sample(n=min(n_neg, len(neg_meta)), random_state=42)
    df_lite = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=42)

    sample_indices = df_lite.index.values.astype(np.int64)

    # 3. Extract vectors from full FAISS index
    # IndexFlatIP reconstructs vectors
    vectors_lite = np.zeros((n_lite, 64), dtype=np.float32)
    for i, idx in enumerate(sample_indices):
        vectors_lite[i] = full_index.reconstruct(int(idx))

    # 4. Build compact FAISS IndexFlatIP
    lite_index = faiss.IndexFlatIP(64)
    lite_index.add(vectors_lite)

    out_index_path = final_dir / "retriever_faiss_lite.index"
    faiss.write_index(lite_index, str(out_index_path))
    size_index_mb = out_index_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved LITE FAISS index: {out_index_path.name} ({size_index_mb:.2f} MB)")

    # 5. Save compact metadata with downcasted types
    df_lite_reset = df_lite.reset_index(drop=True)
    compact_df = df_lite_reset[["encounter_id", "patient_nbr", "readmitted_30d", "time_in_hospital"]].copy()

    # Downcast
    compact_df["encounter_id"] = compact_df["encounter_id"].astype("int32")
    compact_df["patient_nbr"] = compact_df["patient_nbr"].astype("int32")
    compact_df["readmitted_30d"] = compact_df["readmitted_30d"].astype("int8")
    compact_df["time_in_hospital"] = compact_df["time_in_hospital"].astype("int8")

    out_meta_path = final_dir / "retriever_meta_lite.parquet"
    compact_df.to_parquet(out_meta_path, index=False)
    size_meta_kb = out_meta_path.stat().st_size / 1024
    print(f"  ✓ Saved LITE metadata:    {out_meta_path.name} ({size_meta_kb:.1f} KB)")

    # 6. Save LITE cohort scored replay (top 3,000 encounters)
    cohort_path = ROOT_DIR / "reports" / "demo" / "cohort_scored.parquet"
    if cohort_path.exists():
        df_cohort = pd.read_parquet(cohort_path)
        cohort_lite = df_cohort.head(3000).copy()
        out_cohort_path = ROOT_DIR / "reports" / "demo" / "cohort_scored_lite.parquet"
        cohort_lite.to_parquet(out_cohort_path, index=False)
        size_cohort_kb = out_cohort_path.stat().st_size / 1024
        print(f"  ✓ Saved LITE cohort:      {out_cohort_path.name} ({size_cohort_kb:.1f} KB)")

    print("-" * 65)
    print("[SUCCESS] LITE serving artifacts created successfully.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    generate_lite_artifacts()
