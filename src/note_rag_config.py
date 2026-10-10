"""Shared constants for the note-embedding RAG feature (paths, k grid, seeds)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR, YAML_CFG

SPLITS: Tuple[str, ...] = ("train", "val", "test")
MODEL_KEYS: Tuple[str, ...] = ("jina_v5_small", "minilm", "bge_small")
VARIANTS: Tuple[str, ...] = ("v1", "v2")

# READMITIQ_PROCESSED_DIR lets a dry run on synthetic data stay away from data/processed.
PROCESSED_DIR: Path = Path(os.getenv("READMITIQ_PROCESSED_DIR", str(PROCESSED_DATA_DIR)))

NOTE_RAG_DATA_DIR: Path = PROCESSED_DIR / "note_rag"
NOTE_RAG_REPORT_DIR: Path = REPORTS_DIR / "note_rag"
NOTE_RAG_FIGURES_DIR: Path = NOTE_RAG_REPORT_DIR / "figures"
NOTE_RAG_COLAB_DIR: Path = NOTE_RAG_REPORT_DIR / "colab"

MANIFEST_NAME = "MANIFEST.json"
SEED: int = RANDOM_SEED
N_FOLDS: int = int(YAML_CFG.get("rag", {}).get("oof_splits", 5))

# The k grid is shared by every split; the largest value is the only one searched.
K_GRID: Tuple[int, ...] = (5, 10, 20, 50)
K_PRIMARY = 20
SIM_DECIMALS = 8  # similarities are rounded to this many decimals only to break ties

FEATURE_NAMES: Tuple[str, ...] = (
    "rag_note_rate_k5",
    "rag_note_rate_k10",
    "rag_note_rate_k20",
    "rag_note_rate_k50",
    "rag_note_wrate_k20",
    "rag_note_sim_top1",
    "rag_note_sim_mean_k20",
)
PRIMARY_FEATURE = "rag_note_rate_k20"

LABEL_COL = "readmitted_30d"
PATIENT_COL = "patient_nbr"
ID_COL = "encounter_id"


def notes_v2_path(split: str, data_dir: Path = NOTE_RAG_DATA_DIR) -> Path:
    return data_dir / f"notes_v2_{split}.parquet"


def features_path(split: str, data_dir: Path = NOTE_RAG_DATA_DIR) -> Path:
    return data_dir / f"note_rag_features_{split}.parquet"


def split_parquet(split: str) -> Path:
    return PROCESSED_DIR / f"{split}.parquet"


def embedding_name(model_key: str, variant: str, split: str) -> str:
    return f"embeddings_{model_key}_{variant}_{split}.npy"


def index_name(model_key: str, variant: str) -> str:
    return f"embeddings_{model_key}_{variant}_index.parquet"


def expected_split_rows() -> Dict[str, int]:
    """Row count per split, read from the saved feature metadata (never hardcoded)."""
    import json

    meta = json.loads((PROCESSED_DIR / "feature_metadata.json").read_text(encoding="utf-8"))
    return {name: int(info["rows"]) for name, info in meta["splits"].items()}
