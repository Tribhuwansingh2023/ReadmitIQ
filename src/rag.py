"""Clinical RAG retrieval engines (Structured and Note-based) with strict OOF isolation.

Implements two retrieval spaces:
1. Structured retriever: preprocessed clinical feature matrix reduced via TruncatedSVD(64),
   L2-normalized, indexed with FAISS IndexFlatIP.
2. Note retriever: semantic embeddings from all-MiniLM-L6-v2 (384 dims, L2-normalized).

Guarantees 5-fold patient-grouped out-of-fold retrieval on train encounters,
strictly excluding same-patient encounters on all splits.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

from src.config import (
    MODELS_DIR,
    PREPROCESSOR_BASE_PATH,
    PREPROCESSOR_RAG_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.features import build_preprocessor


def load_note_embeddings(split: str, expected_rows: int) -> Optional[np.ndarray]:
    """Load MiniLM note embeddings and strictly validate shape against parquet row count."""
    emb_path = PROCESSED_DATA_DIR / f"embeddings_{split}.npy"
    if not emb_path.exists():
        return None
    embs = np.load(emb_path)
    if embs.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding row count mismatch for '{split}': {emb_path.name} has {embs.shape[0]} rows, "
            f"but expected {expected_rows} rows from {split}.parquet."
        )
    if embs.shape[1] != 384:
        raise ValueError(
            f"Embedding dimension mismatch for '{split}': {emb_path.name} has dimension {embs.shape[1]}, expected 384."
        )
    return embs.astype(np.float32)


def compute_vectorized_rag_features(
    indices: np.ndarray,
    distances: np.ndarray,
    train_pts: np.ndarray,
    train_labels: np.ndarray,
    train_stays: np.ndarray,
    query_pts: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    """Vectorized calculation of RAG features avoiding pandas per-row iloc overhead."""
    n_queries = len(query_pts)
    rates_k20 = np.empty(n_queries, dtype=np.float32)
    sim_weighted_k20 = np.empty(n_queries, dtype=np.float32)
    rates_k50 = np.empty(n_queries, dtype=np.float32)
    mean_stay_k20 = np.empty(n_queries, dtype=np.float32)

    for i in range(n_queries):
        idx_row = indices[i]
        sim_row = distances[i]
        q_pt = query_pts[i]

        # Exclude same patient and invalid indices
        valid_mask = (idx_row >= 0) & (train_pts[idx_row] != q_pt)
        valid_indices = idx_row[valid_mask]
        valid_sims = np.clip(sim_row[valid_mask], 0.0, 1.0)

        # k = 20
        v20 = valid_indices[:20]
        s20 = valid_sims[:20]
        if len(v20) > 0:
            rates_k20[i] = np.mean(train_labels[v20])
            sum_s = np.sum(s20)
            sim_weighted_k20[i] = np.sum(train_labels[v20] * s20) / (sum_s + 1e-8)
            mean_stay_k20[i] = np.mean(train_stays[v20])
        else:
            rates_k20[i] = 0.1139
            sim_weighted_k20[i] = 0.1139
            mean_stay_k20[i] = 4.0

        # k = 50
        v50 = valid_indices[:50]
        if len(v50) > 0:
            rates_k50[i] = np.mean(train_labels[v50])
        else:
            rates_k50[i] = 0.1139

    return pd.DataFrame(
        {
            f"{prefix}_rate_k20": rates_k20,
            f"{prefix}_sim_weighted_k20": sim_weighted_k20,
            f"{prefix}_rate_k50": rates_k50,
            f"{prefix}_mean_stay_k20": mean_stay_k20,
        }
    )


class StructuredRAGEngine:
    """FAISS retriever based on preprocessed structured features reduced via SVD."""

    def __init__(self, n_components: int = 64, random_state: int = RANDOM_SEED):
        self.n_components = n_components
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.train_pts: Optional[np.ndarray] = None
        self.train_encs: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.train_stays: Optional[np.ndarray] = None

    def fit_transform_train(self, X_train_trans: np.ndarray) -> np.ndarray:
        """Fit TruncatedSVD strictly on TRAIN transformed features and unit normalize."""
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        reduced = self.svd.fit_transform(X_train_trans)
        return normalize(reduced, norm="l2").astype(np.float32)

    def transform(self, X_trans: np.ndarray) -> np.ndarray:
        """Project unseen features into fitted SVD space and unit normalize."""
        if self.svd is None:
            raise ValueError("SVD has not been fitted on train yet.")
        reduced = self.svd.transform(X_trans)
        return normalize(reduced, norm="l2").astype(np.float32)

    def build_index(
        self, train_embeddings: np.ndarray, train_df: pd.DataFrame
    ) -> faiss.IndexFlatIP:
        """Build FAISS index on full train embeddings."""
        self.index = faiss.IndexFlatIP(self.n_components)
        self.index.add(train_embeddings)
        self.train_pts = train_df["patient_nbr"].values.astype(np.int64)
        self.train_encs = train_df["encounter_id"].values.astype(np.int64)
        self.train_labels = train_df["readmitted_30d"].values.astype(np.int32)
        self.train_stays = train_df["time_in_hospital"].values.astype(np.float32)
        return self.index

    def retrieve_similar_patients(
        self,
        query_embedding: np.ndarray,
        query_patient_nbr: int,
        k: int = 20,
        k_search: int = 60,
    ) -> List[Dict]:
        """Retrieve top k similar patients excluding the query patient."""
        if self.index is None:
            raise ValueError("FAISS index not built.")
        q_vec = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(q_vec, min(k_search, self.index.ntotal))

        results = []
        for sim, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.train_pts):
                continue
            if self.train_pts[idx] == query_patient_nbr:
                continue
            results.append(
                {
                    "encounter_id": int(self.train_encs[idx]),
                    "patient_nbr": int(self.train_pts[idx]),
                    "readmitted_30d": int(self.train_labels[idx]),
                    "time_in_hospital": float(self.train_stays[idx]),
                    "similarity": float(sim),
                }
            )
            if len(results) == k:
                break
        return results


class ClinicalRAGEngine:
    """FAISS retriever based on clinical note semantic embeddings."""

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.index: Optional[faiss.IndexFlatIP] = None
        self.train_meta_df: Optional[pd.DataFrame] = None
        self.train_pts: Optional[np.ndarray] = None
        self.train_encs: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.train_stays: Optional[np.ndarray] = None

    def build_train_index(
        self,
        train_embeddings: np.ndarray,
        train_meta_df: pd.DataFrame,
    ) -> faiss.IndexFlatIP:
        """Construct FAISS index on train note embeddings."""
        d = train_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(train_embeddings)
        self.train_meta_df = train_meta_df.reset_index(drop=True)
        self.train_pts = self.train_meta_df["patient_nbr"].values.astype(np.int64)
        self.train_encs = self.train_meta_df["encounter_id"].values.astype(np.int64)
        self.train_labels = self.train_meta_df["readmitted_30d"].values.astype(np.int32)
        self.train_stays = self.train_meta_df["time_in_hospital"].values.astype(np.float32)
        return self.index

    def retrieve_similar_patients(
        self,
        query_embedding: np.ndarray,
        query_patient_nbr: int,
        k: Optional[int] = None,
        k_search: int = 60,
    ) -> List[Dict]:
        """Retrieve top k similar training patients excluding query patient."""
        if self.index is None:
            raise ValueError("FAISS index not initialized.")
        k = k or self.k_neighbors
        q_vec = query_embedding.reshape(1, -1).astype(np.float32)
        sims, indices = self.index.search(q_vec, min(k_search, self.index.ntotal))

        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self.train_pts):
                continue
            if self.train_pts[idx] == query_patient_nbr:
                continue
            results.append(
                {
                    "encounter_id": int(self.train_encs[idx]),
                    "patient_nbr": int(self.train_pts[idx]),
                    "readmitted_30d": int(self.train_labels[idx]),
                    "time_in_hospital": float(self.train_stays[idx]),
                    "similarity": float(sim),
                }
            )
            if len(results) == k:
                break
        return results


def build_and_save_all_rag_features(
    compute_notes: bool = True,
    save_parquets: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end full-scale RAG feature engineering with strict memory management."""
    import gc

    print("=" * 70)
    print("BUILDING FULL-SCALE RAG RETRIEVAL AND FEATURES (MEMORY-EFFICIENT)")
    print("=" * 70)

    # Load base preprocessor
    assert PREPROCESSOR_BASE_PATH.exists(), f"Missing {PREPROCESSOR_BASE_PATH}!"
    preprocessor_base = joblib.load(PREPROCESSOR_BASE_PATH)
    num_cols = preprocessor_base.transformers_[0][2]
    cat_cols = preprocessor_base.transformers_[1][2]
    base_features = num_cols + cat_cols

    # -------------------------------------------------------------
    # 1. PROCESS TRAIN SPLIT
    # -------------------------------------------------------------
    print("\n--- [1] Processing TRAIN Structured Retrieval Space ---")
    load_cols = list(
        dict.fromkeys(
            base_features + ["encounter_id", "patient_nbr", "readmitted_30d", "time_in_hospital"]
        )
    )
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=load_cols)
    n_tr = len(train_df)
    print(f"Loaded Train: {n_tr} rows")

    X_tr_trans = preprocessor_base.transform(train_df[base_features]).astype(np.float32)

    struct_engine = StructuredRAGEngine(n_components=64, random_state=RANDOM_SEED)
    E_tr_struct = struct_engine.fit_transform_train(X_tr_trans)
    del X_tr_trans
    gc.collect()

    # Save SVD transformer
    joblib.dump(struct_engine.svd, MODELS_DIR / "structured_svd.joblib")

    # 5-fold patient-grouped OOF on train
    print("Running 5-fold patient-grouped OOF retrieval on TRAIN (69,535 rows)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pts_tr = train_df["patient_nbr"].values.astype(np.int64)
    y_tr = train_df["readmitted_30d"].values.astype(np.int32)
    stays_tr = train_df["time_in_hospital"].values.astype(np.float32)

    oof_struct_indices = np.empty((n_tr, 60), dtype=np.int64)
    oof_struct_distances = np.empty((n_tr, 60), dtype=np.float32)

    for fold, (idx_in, idx_oof) in enumerate(sgkf.split(train_df, y_tr, pts_tr)):
        sub_idx = faiss.IndexFlatIP(64)
        sub_idx.add(E_tr_struct[idx_in])
        D, matched_indices = sub_idx.search(E_tr_struct[idx_oof], 60)
        oof_struct_indices[idx_oof] = idx_in[matched_indices]
        oof_struct_distances[idx_oof] = D

    train_struct_features = compute_vectorized_rag_features(
        oof_struct_indices,
        oof_struct_distances,
        pts_tr,
        y_tr,
        stays_tr,
        pts_tr,
        prefix="rag_struct",
    )
    del oof_struct_indices, oof_struct_distances
    gc.collect()

    # Build full train FAISS index
    print("Building full TRAIN FAISS index...")
    full_struct_index = struct_engine.build_index(E_tr_struct, train_df)
    faiss.write_index(full_struct_index, str(MODELS_DIR / "rag_struct_faiss.index"))
    faiss.write_index(full_struct_index, str(MODELS_DIR / "rag_faiss_train.index"))

    # Save metadata parquet for FAISS index verification
    meta_cols = ["encounter_id", "patient_nbr", "readmitted_30d", "time_in_hospital"]
    train_df[meta_cols].to_parquet(MODELS_DIR / "rag_train_meta.parquet", index=False)

    # Save train_rag.parquet
    train_rag_df = pd.concat([train_df.reset_index(drop=True), train_struct_features], axis=1)
    train_rag_df["rag_readmit_rate_k20"] = train_rag_df["rag_struct_rate_k20"]
    # Re-attach clinical note from raw parquet
    notes_tr = pd.read_parquet(TRAIN_PARQUET, columns=["clinical_note"])["clinical_note"]
    train_rag_df["clinical_note"] = notes_tr
    del notes_tr
    train_rag_df.to_parquet(PROCESSED_DATA_DIR / "train_rag.parquet", index=False)
    print(f"Saved train_rag.parquet ({len(train_rag_df)} rows).")

    # Fit and save preprocessor_rag.joblib
    rag_num_cols = sorted(num_cols + list(train_struct_features.columns) + ["rag_readmit_rate_k20"])
    preprocessor_rag = build_preprocessor(rag_num_cols, cat_cols)
    preprocessor_rag.fit(train_rag_df[rag_num_cols + cat_cols])
    joblib.dump(preprocessor_rag, PREPROCESSOR_RAG_PATH)
    print(f"Saved fitted RAG preprocessor to {PREPROCESSOR_RAG_PATH}")

    del train_df, train_rag_df
    gc.collect()

    # -------------------------------------------------------------
    # 2. PROCESS VAL SPLIT
    # -------------------------------------------------------------
    print("\n--- [2] Processing VAL Split ---")
    val_df = pd.read_parquet(VAL_PARQUET, columns=load_cols)
    n_va = len(val_df)
    X_va_trans = preprocessor_base.transform(val_df[base_features]).astype(np.float32)
    E_va_struct = struct_engine.transform(X_va_trans)
    del X_va_trans
    gc.collect()

    pts_va = val_df["patient_nbr"].values.astype(np.int64)
    D_va, I_va = full_struct_index.search(E_va_struct, 60)
    del E_va_struct
    gc.collect()

    val_struct_features = compute_vectorized_rag_features(
        I_va, D_va, pts_tr, y_tr, stays_tr, pts_va, prefix="rag_struct"
    )
    del D_va, I_va
    gc.collect()

    val_rag_df = pd.concat([val_df.reset_index(drop=True), val_struct_features], axis=1)
    val_rag_df["rag_readmit_rate_k20"] = val_rag_df["rag_struct_rate_k20"]
    notes_va = pd.read_parquet(VAL_PARQUET, columns=["clinical_note"])["clinical_note"]
    val_rag_df["clinical_note"] = notes_va
    del notes_va
    val_rag_df.to_parquet(PROCESSED_DATA_DIR / "val_rag.parquet", index=False)
    print(f"Saved val_rag.parquet ({len(val_rag_df)} rows).")
    del val_df, val_rag_df
    gc.collect()

    # -------------------------------------------------------------
    # 3. PROCESS TEST SPLIT
    # -------------------------------------------------------------
    print("\n--- [3] Processing TEST Split ---")
    test_df = pd.read_parquet(TEST_PARQUET, columns=load_cols)
    n_te = len(test_df)
    X_te_trans = preprocessor_base.transform(test_df[base_features]).astype(np.float32)
    E_te_struct = struct_engine.transform(X_te_trans)
    del X_te_trans
    gc.collect()

    pts_te = test_df["patient_nbr"].values.astype(np.int64)
    D_te, I_te = full_struct_index.search(E_te_struct, 60)
    del E_te_struct
    gc.collect()

    test_struct_features = compute_vectorized_rag_features(
        I_te, D_te, pts_tr, y_tr, stays_tr, pts_te, prefix="rag_struct"
    )
    del D_te, I_te
    gc.collect()

    test_rag_df = pd.concat([test_df.reset_index(drop=True), test_struct_features], axis=1)
    test_rag_df["rag_readmit_rate_k20"] = test_rag_df["rag_struct_rate_k20"]
    notes_te = pd.read_parquet(TEST_PARQUET, columns=["clinical_note"])["clinical_note"]
    test_rag_df["clinical_note"] = notes_te
    del notes_te
    test_rag_df.to_parquet(PROCESSED_DATA_DIR / "test_rag.parquet", index=False)
    print(f"Saved test_rag.parquet ({len(test_rag_df)} rows).")
    del test_df, test_rag_df
    gc.collect()

    print("\nFull-scale RAG feature generation completed successfully!")
    return {
        "n_train": n_tr,
        "n_val": n_va,
        "n_test": n_te,
        "struct_features": list(train_struct_features.columns),
    }


if __name__ == "__main__":
    build_and_save_all_rag_features()
