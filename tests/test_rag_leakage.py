import re
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import pytest

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.notes import generate_single_clinical_note
from src.rag import ClinicalRAGEngine


@pytest.fixture(scope="module")
def rag_artifacts():
    train_rag_path = PROCESSED_DATA_DIR / "train_rag.parquet"
    val_rag_path = PROCESSED_DATA_DIR / "val_rag.parquet"
    test_rag_path = PROCESSED_DATA_DIR / "test_rag.parquet"
    index_path = MODELS_DIR / "rag_faiss_train.index"
    meta_path = MODELS_DIR / "rag_train_meta.parquet"

    assert train_rag_path.exists(), "train_rag.parquet not found"
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


def test_notes_contain_no_label_leakage(rag_artifacts):
    """Test that generated clinical notes contain zero hint of readmission labels or targets."""
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
        assert "clinical_note" in df.columns, f"Missing clinical_note column in {name}"
        # Inspect notes across split
        for note in df["clinical_note"].iloc[:500]:
            for pat in forbidden_patterns:
                assert not pat.search(note), f"Label leakage pattern '{pat.pattern}' found in {name} note:\n{note}"


def test_faiss_index_contains_train_patients_only(rag_artifacts):
    """Test that the FAISS index indexes ONLY train encounters and zero val/test patients."""
    meta_df = rag_artifacts["meta_df"]
    val_df = rag_artifacts["val_df"]
    test_df = rag_artifacts["test_df"]
    index = rag_artifacts["index"]

    assert index.ntotal == len(meta_df), "Mismatch between FAISS total and metadata rows"

    indexed_patients = set(meta_df["patient_nbr"])
    val_patients = set(val_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])

    # Strict isolation: 0 val or test patients may exist in FAISS index
    assert len(indexed_patients.intersection(val_patients)) == 0, "Val patients leaked into FAISS index!"
    assert len(indexed_patients.intersection(test_patients)) == 0, "Test patients leaked into FAISS index!"


def test_retrieval_strictly_excludes_self_and_same_patient(rag_artifacts):
    """Hard Rule 3 Test: Verify that a patient NEVER retrieves themselves or any encounter of the same patient."""
    meta_df = rag_artifacts["meta_df"]
    index = rag_artifacts["index"]

    engine = ClinicalRAGEngine(k_neighbors=20)
    engine.index = index
    engine.train_meta_df = meta_df

    # Load saved train embeddings
    train_embs = np.load(PROCESSED_DATA_DIR / "embeddings_train.npy")

    # Sample 100 queries from train set
    np.random.seed(42)
    sample_indices = np.random.choice(len(meta_df), size=min(100, len(meta_df)), replace=False)

    for idx in sample_indices:
        query_row = meta_df.iloc[idx]
        query_emb = train_embs[idx]
        query_pt = int(query_row["patient_nbr"])
        query_enc = int(query_row["encounter_id"])

        neighbors = engine.retrieve_similar_patients(
            query_embedding=query_emb,
            query_patient_nbr=query_pt,
            k=20,
            k_search=50,
        )

        assert len(neighbors) == 20, f"Expected 20 neighbors, got {len(neighbors)}"

        retrieved_patients = [n["patient_nbr"] for n in neighbors]
        retrieved_encounters = [n["encounter_id"] for n in neighbors]

        # PROOF: Patient never retrieves themselves
        assert query_enc not in retrieved_encounters, f"Encounter {query_enc} retrieved itself!"
        # PROOF: Patient never retrieves ANY encounter of the same patient
        assert query_pt not in retrieved_patients, f"Patient {query_pt} retrieved an encounter of themselves!"


def test_rag_feature_bounds_and_validity(rag_artifacts):
    """Test that rag_readmit_rate_k20 is a valid probability with no missing values."""
    for name in ["train_df", "val_df", "test_df"]:
        df = rag_artifacts[name]
        assert "rag_readmit_rate_k20" in df.columns, f"Missing rag_readmit_rate_k20 in {name}"
        col = df["rag_readmit_rate_k20"]

        assert col.isnull().sum() == 0, f"NaNs found in {name} rag_readmit_rate_k20"
        assert (col >= 0.0).all(), f"Values below 0 in {name} rag_readmit_rate_k20"
        assert (col <= 1.0).all(), f"Values above 1 in {name} rag_readmit_rate_k20"

        # Mean rate should be close to empirical target rate (~11%)
        mean_rate = col.mean()
        assert 0.05 <= mean_rate <= 0.25, f"Unreasonable RAG mean rate in {name}: {mean_rate:.4f}"
