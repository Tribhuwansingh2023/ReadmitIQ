"""Loading split metadata for note-RAG scripts, with the test split label-free by design."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.note_rag_config import ID_COL, LABEL_COL, PATIENT_COL, split_parquet


def load_split_meta(split: str, columns: Optional[list] = None) -> pd.DataFrame:
    """Read encounter id, patient and (except for test) the label from a cleaned split parquet.

    The test split is only ever read with id and patient columns; asking for the label of the
    test split raises, so no test label can enter any note-RAG computation.
    """
    wanted = [ID_COL, PATIENT_COL] + ([LABEL_COL] if split != "test" else [])
    if columns:
        wanted = list(dict.fromkeys(wanted + columns))
    if split == "test" and LABEL_COL in wanted:
        raise PermissionError("The test split label must never be read in note-RAG code")
    return pd.read_parquet(split_parquet(split), columns=wanted)


def meta_arrays(meta: pd.DataFrame) -> dict:
    """Numpy views of the id, patient and (when present) label columns."""
    arrays = {
        "ids": meta[ID_COL].to_numpy(),
        "patients": meta[PATIENT_COL].to_numpy().astype(np.int64),
    }
    if LABEL_COL in meta.columns:
        arrays["y"] = meta[LABEL_COL].to_numpy().astype(np.int64)
    return arrays
