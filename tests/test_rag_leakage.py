import re
from pathlib import Path
import faiss
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    MODELS_DIR,
    PREPROCESSOR_BASE_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.rag import StructuredRAGEngine


@pytest.fixture(scope="module")
def rag_artifacts():
    train_rag_path = PROCESSED_DATA_DIR / "train_rag.parquet"
    val_rag_path = PROCESSED_DATA_DIR / "val_rag.parquet"
    test_rag_path = PROCESSED_DATA_DIR / "test_rag.parquet"
    index_path = MODELS_DIR / "rag_faiss_train.index"
    meta_path = MODELS_DIR / "rag_train_meta.parquet"

    assert train_rag_path.exists(), "train_rag.parquet not found"
    assert val_rag_path.exists(), "val_rag.parquet not found"
    assert test_rag_path.exists(), "test_rag.parquet not found"
    assert index_path.exists(), "rag_faiss_train.index not found"
    assert meta_path.exists(), "rag_train_meta.parquet not found"

    train_df = pd.read_parquet(train_rag_path)
    val_df = pd.read_parquet(val_rag_path)
    test_df = pd.read_parquet(test_rag_path)
    meta_df = pd.read_parquet(meta_path)
    index = faiss.read_index(str(index_path))

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "meta_df": meta_df,
        "index": index,
    }


def test_notes_contain_no_label_leakage_full_corpus(rag_artifacts):
    """Test that generated clinical notes contain zero hint of readmission labels across ALL rows."""
    train_df = rag_artifacts["train_df"]
    val_df = rag_artifacts["val_df"]
    test_df = rag_artifacts["test_df"]

    forbidden_patterns = [
        re.compile(r"\breadmit\b", re.IGNORECASE),
        re.compile(r"\breadmission\b", re.IGNORECASE),
        re.compile(r"\breadmitted\b", re.IGNORECASE),
        re.compile(r"(?<!\d)<30(?!\d)"),
        re.compile(r"(?<!\d)>30(?!\d)"),
        re.compile(r"\boutcome\b", re.IGNORECASE),
        re.compile(r"\brecurrent\b", re.IGNORECASE),
    ]

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "clinical_note" in df.columns, f"Missing clinical_note in {name}"
        notes = df["clinical_note"].tolist()
        for idx, note in enumerate(notes):
            for pat in forbidden_patterns:
                assert not pat.search(note), (
                    f"Label leakage '{pat.pattern}' found in {name} row {idx}: {note}"
                )


def test_faiss_index_full_train_size_and_disjoint_patients(rag_artifacts):
    """Test that the FAISS index indexes ALL 69,535 train encounters and zero val/test patients."""
    meta_df = rag_artifacts["meta_df"]
    val_df = rag_artifacts["val_df"]
    test_df = rag_artifacts["test_df"]
    index = rag_artifacts["index"]

    assert index.ntotal == len(meta_df), "Mismatch between FAISS total and metadata rows"
    assert index.ntotal == 69535, f"Expected 69,535 full train rows, got {index.ntotal}"

    indexed_patients = set(meta_df["patient_nbr"])
    val_patients = set(val_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])

    assert len(indexed_patients.intersection(val_patients)) == 0, "Val patients leaked into FAISS index!"
    assert len(indexed_patients.intersection(test_patients)) == 0, "Test patients leaked into FAISS index!"


def test_oof_folds_are_patient_disjoint():
    """Verify that train OOF folds are strictly patient-disjoint."""
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["patient_nbr", "readmitted_30d"])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    groups = train_df["patient_nbr"].values
    y = train_df["readmitted_30d"].values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups)):
        train_pts = set(groups[train_idx])
        val_pts = set(groups[val_idx])
        overlap = train_pts.intersection(val_pts)
        assert len(overlap) == 0, f"OOF fold {fold} has {len(overlap)} overlapping patients!"


def test_retrieval_strictly_excludes_self_and_same_patient_all_splits(rag_artifacts):
    """Verify that querying the FAISS index never retrieves any encounter of the same patient."""
    meta_df = rag_artifacts["meta_df"]
    val_df = rag_artifacts["val_df"]
    test_df = rag_artifacts["test_df"]
    index = rag_artifacts["index"]

    svd = joblib.load(MODELS_DIR / "structured_svd.joblib")
    preprocessor = joblib.load(PREPROCESSOR_BASE_PATH)
    base_features = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]

    engine = StructuredRAGEngine(n_components=64)
    engine.svd = svd
    engine.index = index
    engine.train_pts = meta_df["patient_nbr"].values.astype(np.int64)
    engine.train_encs = meta_df["encounter_id"].values.astype(np.int64)
    engine.train_labels = meta_df["readmitted_30d"].values.astype(np.int32)
    engine.train_stays = meta_df["time_in_hospital"].values.astype(np.float32)

    train_df = rag_artifacts["train_df"]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sub_sample = df.sample(min(50, len(df)), random_state=42)
        X_trans = preprocessor.transform(sub_sample[base_features])
        embs = engine.transform(X_trans)

        for i, (_, row) in enumerate(sub_sample.iterrows()):
            q_pt = int(row["patient_nbr"])
            q_enc = int(row["encounter_id"])
            q_emb = embs[i]

            neighbors = engine.retrieve_similar_patients(
                query_embedding=q_emb,
                query_patient_nbr=q_pt,
                k=20,
                k_search=60,
            )

            assert len(neighbors) == 20, f"Expected 20 neighbors for {name} row, got {len(neighbors)}"
            for nbr in neighbors:
                assert nbr["patient_nbr"] != q_pt, f"Retrieved encounter of same patient {q_pt} in {name}!"
                assert nbr["encounter_id"] != q_enc, f"Retrieved identical encounter {q_enc} in {name}!"


def test_train_val_rate_distributions_same_k(rag_artifacts):
    """Test that train and validation RAG features use the exact same k=20 denominator."""
    train_df = rag_artifacts["train_df"]
    val_df = rag_artifacts["val_df"]

    col = "rag_struct_rate_k20"
    assert col in train_df.columns, f"Missing {col} in train_df"
    assert col in val_df.columns, f"Missing {col} in val_df"

    # Distinct values must be multiples of 0.05 (1/20)
    for name, df in [("train", train_df), ("val", val_df)]:
        unique_vals = np.round(df[col].unique(), 4)
        for v in unique_vals:
            remainder = round(v * 20.0, 3) % 1.0
            assert remainder < 1e-3 or remainder > 0.999, (
                f"Non-k=20 fraction found in {name} {col}: {v}"
            )
