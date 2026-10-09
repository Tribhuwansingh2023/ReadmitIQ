#!/usr/bin/env python3
"""Generate ultra-lightweight CI fixtures (<350 KB total) for GitHub Actions and clean clones.

Uses real preprocessors and models, pairing them with micro FAISS and cohort datasets:
- models/final/model.joblib (real lightweight LightGBM, 275 KB)
- models/final/calibrator.joblib (real calibrator, 2.3 KB)
- models/final/preprocessor_base.joblib & preprocessor.joblib (real pipelines, 23 KB)
- models/final/structured_svd.joblib (real SVD, 36 KB)
- models/final/retriever_faiss.index & retriever_faiss_lite.index (50 vectors x 64d, 15 KB)
- models/final/retriever_meta.parquet & retriever_meta_lite.parquet (50 rows, 5 KB)
- models/final/feature_list.json & models/FROZEN_SELECTION.json (3 KB)
- reports/demo/cohort_scored.parquet & cohort_scored_lite.parquet (50 rows, 5 KB)
"""

import io
import json
import shutil
import sys
from pathlib import Path

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import faiss
import joblib
import numpy as np
import pandas as pd


def generate_ci_fixtures():
    print("=" * 65)
    print("🛠️ Generating Lightweight ReadmitIQ CI Fixtures (< 400 KB)")
    print("=" * 65)

    real_final = ROOT_DIR / "models" / "final"
    fixtures_dir = ROOT_DIR / "tests" / "fixtures"
    fix_models_final = fixtures_dir / "models" / "final"
    fix_reports_demo = fixtures_dir / "reports" / "demo"
    fix_reports_metrics = fixtures_dir / "reports" / "metrics"

    fix_models_final.mkdir(parents=True, exist_ok=True)
    fix_reports_demo.mkdir(parents=True, exist_ok=True)
    fix_reports_metrics.mkdir(parents=True, exist_ok=True)

    # 1. Copy lightweight joblibs & configs (< 350 KB)
    files_to_copy = [
        "model.joblib",
        "calibrator.joblib",
        "preprocessor_base.joblib",
        "preprocessor.joblib",
        "structured_svd.joblib",
        "feature_list.json",
    ]
    for fname in files_to_copy:
        src = real_final / fname
        if src.exists():
            shutil.copy(src, fix_models_final / fname)

    src_frozen = ROOT_DIR / "models" / "FROZEN_SELECTION.json"
    if src_frozen.exists():
        shutil.copy(src_frozen, fixtures_dir / "models" / "FROZEN_SELECTION.json")

    for m_file in [
        "ablation_table.json",
        "test_results.json",
        "fairness_audit.json",
        "drift_metrics.json",
    ]:
        p = ROOT_DIR / "reports" / "metrics" / m_file
        if p.exists():
            shutil.copy(p, fix_reports_metrics / m_file)

    # 2. Build 50 synthetic encounters matching feature_list
    with open(fix_models_final / "feature_list.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]
    base_num_cols = [c for c in num_cols if c != "rag_struct_rate_k20"]

    n_samples = 50
    np.random.seed(42)
    sample_data = {
        "encounter_id": np.arange(1001, 1001 + n_samples, dtype=np.int32),
        "patient_nbr": np.arange(5001, 5001 + n_samples, dtype=np.int32),
        "readmitted_30d": np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12]).astype(np.int8),
        "age": np.random.choice(["[50-60)", "[60-70)", "[70-80)"], size=n_samples),
        "gender": np.random.choice(["Female", "Male"], size=n_samples),
        "time_in_hospital": np.random.randint(1, 10, size=n_samples, dtype=np.int8),
        "number_inpatient": np.random.randint(0, 4, size=n_samples, dtype=np.int8),
        "diag_1_group": np.random.choice(
            ["Circulatory", "Diabetes", "Respiratory", "Other"], size=n_samples
        ),
    }

    for col in base_num_cols:
        if col not in sample_data:
            sample_data[col] = np.random.uniform(0, 10, size=n_samples)
    for col in cat_cols:
        if col not in sample_data:
            sample_data[col] = "Missing"

    df_synth = pd.DataFrame(sample_data)

    # 3. Transform through real preprocessor_base and real SVD -> exactly 64 dims
    preproc_base = joblib.load(fix_models_final / "preprocessor_base.joblib")
    svd = joblib.load(fix_models_final / "structured_svd.joblib")

    dense_feats = preproc_base.transform(df_synth[base_num_cols + cat_cols])
    if hasattr(dense_feats, "toarray"):
        dense_feats = dense_feats.toarray()

    query_vecs = svd.transform(dense_feats).astype(np.float32)
    faiss.normalize_L2(query_vecs)

    # 4. Save micro FAISS indexes (both standard and lite)
    index = faiss.IndexFlatIP(64)
    index.add(query_vecs)
    faiss.write_index(index, str(fix_models_final / "retriever_faiss.index"))
    faiss.write_index(index, str(fix_models_final / "retriever_faiss_lite.index"))

    # 5. Save micro retriever metadata
    retriever_meta = df_synth[
        [
            "encounter_id",
            "patient_nbr",
            "readmitted_30d",
            "time_in_hospital",
            "age",
            "gender",
            "number_inpatient",
            "diag_1_group",
        ]
    ].copy()
    retriever_meta.to_parquet(fix_models_final / "retriever_meta.parquet", index=False)
    retriever_meta.to_parquet(fix_models_final / "retriever_meta_lite.parquet", index=False)

    # 6. Save micro cohort scored dataset
    df_synth["risk_score"] = np.random.uniform(0.05, 0.45, size=n_samples).round(4)
    df_synth["risk_tier"] = np.where(df_synth["risk_score"] > 0.25, "High", "Low")
    cohort_cols = [
        "encounter_id",
        "patient_nbr",
        "risk_score",
        "risk_tier",
        "readmitted_30d",
        "age",
        "gender",
        "time_in_hospital",
        "number_inpatient",
        "diag_1_group",
    ]
    cohort_df = df_synth[cohort_cols].sort_values(by="risk_score", ascending=False)
    cohort_df.to_parquet(fix_reports_demo / "cohort_scored.parquet", index=False)
    cohort_df.to_parquet(fix_reports_demo / "cohort_scored_lite.parquet", index=False)

    print("  ✓ Created fixture models in tests/fixtures/models/final/")
    print("  ✓ Created fixture reports in tests/fixtures/reports/")
    print("[SUCCESS] CI fixtures created successfully.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    generate_ci_fixtures()
