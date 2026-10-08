import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    FEATURE_METADATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.notes import generate_notes_for_dataframe


class ClinicalRAGEngine:
    """FAISS-based clinical vector retriever with strict out-of-fold leakage prevention."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k_neighbors: int = 20,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.k_neighbors = k_neighbors
        self.device = device
        self._model = None
        self.index = None
        self.train_meta_df = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading SentenceTransformer ({self.model_name}) on {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_notes(
        self,
        notes: List[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode clinical text notes into unit-normalized dense vectors."""
        embeddings = self.model.encode(
            notes,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build_train_index(
        self,
        train_embeddings: np.ndarray,
        train_meta_df: pd.DataFrame,
    ) -> faiss.IndexFlatIP:
        """Construct FAISS inner product index strictly on TRAIN embeddings."""
        d = train_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(train_embeddings)
        self.index = index
        self.train_meta_df = train_meta_df.reset_index(drop=True)
        return index

    def retrieve_similar_patients(
        self,
        query_embedding: np.ndarray,
        query_patient_nbr: int,
        k: Optional[int] = None,
        k_search: int = 50,
    ) -> List[Dict]:
        """Retrieve top-k similar training patients excluding query patient and encounters."""
        if self.index is None or self.train_meta_df is None:
            raise ValueError("FAISS index has not been initialized.")

        k = k or self.k_neighbors
        q_vec = query_embedding.reshape(1, -1).astype(np.float32)
        sims, indices = self.index.search(q_vec, min(k_search, self.index.ntotal))

        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self.train_meta_df):
                continue
            cand = self.train_meta_df.iloc[idx]
            # Hard Rule: Patient must NEVER retrieve themselves or any encounter of same patient
            if int(cand["patient_nbr"]) == int(query_patient_nbr):
                continue
            results.append(
                {
                    "train_idx": int(idx),
                    "encounter_id": int(cand["encounter_id"]),
                    "patient_nbr": int(cand["patient_nbr"]),
                    "readmitted_30d": int(cand["readmitted_30d"]),
                    "similarity": float(sim),
                }
            )
            if len(results) >= k:
                break

        return results


def compute_leakage_safe_rag_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 20,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ClinicalRAGEngine]:
    """Compute leakage-safe RAG feature for train (via out-of-fold) and val/test."""
    engine = ClinicalRAGEngine(k_neighbors=k)
    dim = train_embeddings.shape[1]

    # Assign 5 disjoint folds to train encounters based on patient_nbr
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_folds = np.zeros(len(train_df), dtype=int)
    y_train = train_df["readmitted_30d"].values
    groups_train = train_df["patient_nbr"].values

    for fold, (_, val_idx) in enumerate(sgkf.split(train_df, y_train, groups_train)):
        train_folds[val_idx] = fold

    train_df_meta = train_df[["encounter_id", "patient_nbr", "readmitted_30d"]].copy()
    train_df_meta["train_fold"] = train_folds

    # 1. TRAIN: Out-Of-Fold RAG retrieval
    print(f"Computing Out-Of-Fold RAG features on TRAIN ({n_splits} folds)...")
    train_rag_rate = np.zeros(len(train_df), dtype=np.float32)

    for fold in range(n_splits):
        cand_mask = train_folds != fold
        query_mask = train_folds == fold

        cand_indices = np.where(cand_mask)[0]
        query_indices = np.where(query_mask)[0]

        cand_embs = train_embeddings[cand_indices]
        cand_meta = train_df_meta.iloc[cand_indices].reset_index(drop=True)

        fold_index = faiss.IndexFlatIP(dim)
        fold_index.add(cand_embs)

        # Search for queries in this fold
        q_embs = train_embeddings[query_indices]
        sims, idxs = fold_index.search(q_embs, min(50, len(cand_indices)))

        for i, q_idx in enumerate(query_indices):
            q_pt = groups_train[q_idx]
            retrieved_labels = []

            for cand_pos in idxs[i]:
                if cand_pos < 0 or cand_pos >= len(cand_meta):
                    continue
                cand_row = cand_meta.iloc[cand_pos]
                # Double-check: strict exclusion of same patient
                if int(cand_row["patient_nbr"]) == int(q_pt):
                    continue
                retrieved_labels.append(int(cand_row["readmitted_30d"]))
                if len(retrieved_labels) == k:
                    break

            # If fewer than k available (rare), fallback to whatever found or train prior
            if retrieved_labels:
                train_rag_rate[q_idx] = np.mean(retrieved_labels)
            else:
                train_rag_rate[q_idx] = float(np.mean(y_train))

    # 2. Build full TRAIN index for Validation and Test queries
    print("Building full TRAIN FAISS index for evaluation sets...")
    engine.build_train_index(train_embeddings, train_df_meta)

    # 3. VAL: Retrieve against full train index
    print(f"Computing RAG features for VAL ({len(val_df)} rows)...")
    val_rag_rate = np.zeros(len(val_df), dtype=np.float32)
    val_groups = val_df["patient_nbr"].values
    v_sims, v_idxs = engine.index.search(val_embeddings, min(50, engine.index.ntotal))

    for i in range(len(val_df)):
        q_pt = val_groups[i]
        retrieved_labels = []
        for cand_pos in v_idxs[i]:
            if cand_pos < 0 or cand_pos >= len(engine.train_meta_df):
                continue
            cand_row = engine.train_meta_df.iloc[cand_pos]
            if int(cand_row["patient_nbr"]) == int(q_pt):
                continue
            retrieved_labels.append(int(cand_row["readmitted_30d"]))
            if len(retrieved_labels) == k:
                break

        val_rag_rate[i] = np.mean(retrieved_labels) if retrieved_labels else float(np.mean(y_train))

    # 4. TEST: Retrieve against full train index
    print(f"Computing RAG features for TEST ({len(test_df)} rows)...")
    test_rag_rate = np.zeros(len(test_df), dtype=np.float32)
    test_groups = test_df["patient_nbr"].values
    t_sims, t_idxs = engine.index.search(test_embeddings, min(50, engine.index.ntotal))

    for i in range(len(test_df)):
        q_pt = test_groups[i]
        retrieved_labels = []
        for cand_pos in t_idxs[i]:
            if cand_pos < 0 or cand_pos >= len(engine.train_meta_df):
                continue
            cand_row = engine.train_meta_df.iloc[cand_pos]
            if int(cand_row["patient_nbr"]) == int(q_pt):
                continue
            retrieved_labels.append(int(cand_row["readmitted_30d"]))
            if len(retrieved_labels) == k:
                break

        test_rag_rate[i] = np.mean(retrieved_labels) if retrieved_labels else float(np.mean(y_train))

    return train_rag_rate, val_rag_rate, test_rag_rate, engine


def run_rag_pipeline(
    train_parquet: Path = TRAIN_PARQUET,
    val_parquet: Path = VAL_PARQUET,
    test_parquet: Path = TEST_PARQUET,
    output_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    sample_size: Optional[int] = 12000,
    force_recompute: bool = False,
) -> Dict:
    """Run end-to-end notes generation, MiniLM embedding, FAISS indexing and RAG feature engineering."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data splits...")
    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)
    test_df = pd.read_parquet(test_parquet)

    # Subsampling on CPU if requested to ensure fast execution
    if sample_size and sample_size < len(train_df):
        print(f"Subsampling to {sample_size} train rows (stratified by patient and target)...")
        sgkf = StratifiedGroupKFold(n_splits=int(len(train_df) / sample_size), shuffle=True, random_state=RANDOM_SEED)
        train_idx = next(sgkf.split(train_df, train_df["readmitted_30d"], train_df["patient_nbr"]))[1]
        train_df = train_df.iloc[train_idx].reset_index(drop=True)

        # Scale val and test proportionally (approx 20% of train each = ~2400)
        eval_sample = int(sample_size * 0.20)
        sgkf_v = StratifiedGroupKFold(n_splits=max(2, int(len(val_df) / eval_sample)), shuffle=True, random_state=RANDOM_SEED)
        val_idx = next(sgkf_v.split(val_df, val_df["readmitted_30d"], val_df["patient_nbr"]))[1]
        val_df = val_df.iloc[val_idx].reset_index(drop=True)

        sgkf_t = StratifiedGroupKFold(n_splits=max(2, int(len(test_df) / eval_sample)), shuffle=True, random_state=RANDOM_SEED)
        test_idx = next(sgkf_t.split(test_df, test_df["readmitted_30d"], test_df["patient_nbr"]))[1]
        test_df = test_df.iloc[test_idx].reset_index(drop=True)

    print(f"Dataset sizes for RAG pipeline: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 1. Generate clinical notes
    notes_cache = output_dir / ".cache_notes"
    notes_cache.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic clinical notes...")
    train_notes = generate_notes_for_dataframe(train_df)
    val_notes = generate_notes_for_dataframe(val_df)
    test_notes = generate_notes_for_dataframe(test_df)

    train_df["clinical_note"] = train_notes
    val_df["clinical_note"] = val_notes
    test_df["clinical_note"] = test_notes

    # 2. Embed with MiniLM
    engine = ClinicalRAGEngine(k_neighbors=20)
    print("Embedding clinical notes with all-MiniLM-L6-v2...")
    train_embs = engine.encode_notes(train_notes, batch_size=128)
    val_embs = engine.encode_notes(val_notes, batch_size=128)
    test_embs = engine.encode_notes(test_notes, batch_size=128)

    np.save(output_dir / "embeddings_train.npy", train_embs)
    np.save(output_dir / "embeddings_val.npy", val_embs)
    np.save(output_dir / "embeddings_test.npy", test_embs)

    # 3. Compute Out-Of-Fold RAG feature
    train_rag, val_rag, test_rag, trained_engine = compute_leakage_safe_rag_features(
        train_df,
        val_df,
        test_df,
        train_embs,
        val_embs,
        test_embs,
        k=20,
    )

    train_df["rag_readmit_rate_k20"] = train_rag
    val_df["rag_readmit_rate_k20"] = val_rag
    test_df["rag_readmit_rate_k20"] = test_rag

    # 4. Save artifacts
    print("Saving RAG-enhanced parquets and FAISS index...")
    faiss.write_index(trained_engine.index, str(models_dir / "rag_faiss_train.index"))
    trained_engine.train_meta_df.to_parquet(models_dir / "rag_train_meta.parquet", index=False)

    train_df.to_parquet(output_dir / "train_rag.parquet", index=False)
    val_df.to_parquet(output_dir / "val_rag.parquet", index=False)
    test_df.to_parquet(output_dir / "test_rag.parquet", index=False)

    # Update feature metadata to include rag_readmit_rate_k20
    with open(FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)

    if "rag_readmit_rate_k20" not in meta["numeric_features"]:
        meta["numeric_features"].append("rag_readmit_rate_k20")
        meta["numeric_features"].sort()

    with open(FEATURE_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "embedding_dim": int(train_embs.shape[1]),
        "faiss_total_indexed": trained_engine.index.ntotal,
        "rag_train_mean": float(np.mean(train_rag)),
        "rag_val_mean": float(np.mean(val_rag)),
        "rag_test_mean": float(np.mean(test_rag)),
    }
    print("RAG pipeline complete:", summary)
    return summary


if __name__ == "__main__":
    run_rag_pipeline()
