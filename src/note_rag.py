"""Note-embedding retrieval features (RAG-style) with strict leakage protection.

For every encounter we retrieve the most similar past TRAIN encounters (cosine similarity of
note embeddings) and use their historical 30-day readmission rate as a feature.

Leakage rules enforced here and re-checked by assert_* helpers:
    * the index holds TRAIN rows only;
    * TRAIN features are 5-fold patient-grouped out-of-fold (StratifiedGroupKFold, shuffled);
    * validation and test rows query the full TRAIN index;
    * no encounter of the query's own patient is ever returned;
    * only neighbor labels are used, never the query's own label;
    * every split uses the same k grid, and only one search of max(k) is run per row.

Search is exact (float64 dot products) with deterministic tie-breaking: similarities are
rounded to SIM_DECIMALS and ties are ordered by index position.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.note_rag_config import (
    FEATURE_NAMES,
    ID_COL,
    K_GRID,
    K_PRIMARY,
    MANIFEST_NAME,
    N_FOLDS,
    PRIMARY_FEATURE,
    SEED,
    SIM_DECIMALS,
    embedding_name,
    index_name,
)

SEARCH_CHUNK = 256
UNIT_NORM_TOLERANCE = 1e-3


class EmbeddingValidationError(ValueError):
    """Raised when an embeddings file does not match its manifest, index or parquet."""


class LeakageError(AssertionError):
    """Raised when a leakage-protocol check fails."""


# --------------------------------------------------------------------------------------
# Loading and validating embeddings
# --------------------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(emb_dir: Path) -> Dict[str, Any]:
    path = emb_dir / MANIFEST_NAME
    if not path.exists():
        raise EmbeddingValidationError(f"Missing {MANIFEST_NAME} in {emb_dir.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_embedding_array(vecs: np.ndarray, expected_rows: int, name: str) -> None:
    """Shape, dtype, finiteness and unit-norm checks on an in-memory array."""
    if vecs.ndim != 2:
        raise EmbeddingValidationError(f"{name}: expected a 2D array, got {vecs.ndim}D")
    if vecs.shape[0] != expected_rows:
        raise EmbeddingValidationError(
            f"{name}: {vecs.shape[0]} rows but the parquet has {expected_rows}"
        )
    if vecs.dtype != np.float32:
        raise EmbeddingValidationError(f"{name}: dtype {vecs.dtype}, expected float32")
    if not np.isfinite(vecs).all():
        raise EmbeddingValidationError(f"{name}: contains NaN or infinite values")
    norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE):
        raise EmbeddingValidationError(f"{name}: rows are not unit norm")


def validate_index_alignment(
    emb_dir: Path, model_key: str, variant: str, split: str, encounter_ids: np.ndarray
) -> None:
    """The saved index parquet must list exactly the parquet's encounter ids, in order."""
    path = emb_dir / index_name(model_key, variant)
    if not path.exists():
        raise EmbeddingValidationError(f"Missing index parquet {path.name}")
    index = pd.read_parquet(path)
    rows = index[index["split"] == split].sort_values("row")
    if len(rows) != len(encounter_ids):
        raise EmbeddingValidationError(
            f"{path.name}: {len(rows)} {split} rows, parquet has {len(encounter_ids)}"
        )
    if not np.array_equal(rows["row"].to_numpy(), np.arange(len(rows))):
        raise EmbeddingValidationError(f"{path.name}: row numbers are not 0..n-1 for {split}")
    if not np.array_equal(rows[ID_COL].to_numpy(), np.asarray(encounter_ids)):
        raise EmbeddingValidationError(f"{path.name}: encounter order differs from the parquet")


def load_embeddings(
    emb_dir: Path, model_key: str, variant: str, split: str, encounter_ids: np.ndarray
) -> np.ndarray:
    """Load one split's embeddings and validate them against manifest, index and parquet."""
    file_name = embedding_name(model_key, variant, split)
    path = emb_dir / file_name
    if not path.exists():
        raise EmbeddingValidationError(f"Missing embeddings file {file_name}")
    entry = read_manifest(emb_dir)["entries"].get(file_name)
    if entry is None:
        raise EmbeddingValidationError(f"{file_name} is not listed in {MANIFEST_NAME}")
    if entry["sha256"] != sha256_of(path):
        raise EmbeddingValidationError(f"{file_name}: SHA-256 does not match the manifest")
    vecs = np.load(path)
    validate_embedding_array(vecs, len(encounter_ids), file_name)
    if entry["rows"] != vecs.shape[0] or entry["dim"] != vecs.shape[1]:
        raise EmbeddingValidationError(f"{file_name}: shape differs from the manifest")
    validate_index_alignment(emb_dir, model_key, variant, split, encounter_ids)
    return vecs


# --------------------------------------------------------------------------------------
# Exact search that never returns the query's own patient
# --------------------------------------------------------------------------------------
def same_patient_pairs(
    query_patients: np.ndarray, index_patients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (query_row, index_col) pairs that share a patient, fully vectorized."""
    order = np.argsort(index_patients, kind="stable")
    sorted_p = index_patients[order]
    lo = np.searchsorted(sorted_p, query_patients, side="left")
    hi = np.searchsorted(sorted_p, query_patients, side="right")
    counts = hi - lo
    rows = np.repeat(np.arange(len(query_patients)), counts)
    starts = np.repeat(lo - (np.cumsum(counts) - counts), counts)
    cols = order[starts + np.arange(counts.sum())]
    return rows, cols


def exact_search(
    queries: np.ndarray,
    index_vecs: np.ndarray,
    query_patients: np.ndarray,
    index_patients: np.ndarray,
    k: int,
    chunk: int = SEARCH_CHUNK,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-k by cosine similarity (unit vectors), excluding the query's own patient.

    Returns (neighbor positions, similarities), each (n_queries, k). Missing slots, only
    possible when the index has fewer than k eligible rows, hold -1 and NaN.
    """
    n_q, n_idx = len(queries), len(index_vecs)
    out_idx = np.full((n_q, k), -1, dtype=np.int64)
    out_sim = np.full((n_q, k), np.nan, dtype=np.float64)
    index64 = index_vecs.astype(np.float64)
    kk = min(k, n_idx)
    for lo in range(0, n_q, chunk):
        hi = min(lo + chunk, n_q)
        sims = queries[lo:hi].astype(np.float64) @ index64.T
        rows, cols = same_patient_pairs(query_patients[lo:hi], index_patients)
        sims[rows, cols] = -np.inf
        excluded = np.bincount(rows, minlength=hi - lo)
        rounded = np.round(sims, SIM_DECIMALS)
        kth = np.partition(rounded, n_idx - kk, axis=1)[:, n_idx - kk]
        for i in range(hi - lo):
            cand = np.flatnonzero(rounded[i] >= kth[i])
            order = np.lexsort((cand, -rounded[i, cand]))[: min(kk, n_idx - excluded[i])]
            chosen = cand[order]
            out_idx[lo + i, : len(chosen)] = chosen
            out_sim[lo + i, : len(chosen)] = sims[i, chosen]
    return out_idx, out_sim


# --------------------------------------------------------------------------------------
# Out-of-fold folds and features
# --------------------------------------------------------------------------------------
Fold = Tuple[np.ndarray, np.ndarray]


def make_folds(
    y: np.ndarray, patients: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED
) -> List[Fold]:
    """Patient-grouped, label-stratified, shuffled folds: list of (in-fold rows, held-out rows)."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(a, b) for a, b in splitter.split(np.zeros(len(y)), y, patients)]


def oof_neighbors(
    emb: np.ndarray, patients: np.ndarray, folds: Sequence[Fold], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Neighbors for TRAIN rows, each searched only in the other folds (global positions)."""
    idx = np.full((len(emb), k), -1, dtype=np.int64)
    sims = np.full((len(emb), k), np.nan, dtype=np.float64)
    for idx_in, idx_out in folds:
        local, s = exact_search(emb[idx_out], emb[idx_in], patients[idx_out], patients[idx_in], k)
        mapped = np.where(local >= 0, idx_in[np.clip(local, 0, None)], -1)
        idx[idx_out], sims[idx_out] = mapped, s
    return idx, sims


def features_from_neighbors(
    idx: np.ndarray,
    sims: np.ndarray,
    labels: np.ndarray,
    prior: Any,
    k_grid: Sequence[int] = K_GRID,
) -> pd.DataFrame:
    """Turn neighbor positions and similarities into the engineered features (vectorized).

    `labels` are the labels of the indexed rows (never the query's own label). `prior` is the
    rate used only when a row has no neighbor; it is a scalar or one value per row.
    """
    valid = idx >= 0
    lab = np.where(valid, labels[np.clip(idx, 0, None)], 0).astype(np.float64)
    sim = np.where(valid, sims, 0.0)
    cum_lab, cum_cnt = np.cumsum(lab, axis=1), np.cumsum(valid, axis=1)
    prior_arr = np.broadcast_to(np.asarray(prior, dtype=np.float64), (len(idx),))
    cols: Dict[str, np.ndarray] = {}
    for k in k_grid:
        n_k = cum_cnt[:, k - 1]
        cols[f"rag_note_rate_k{k}"] = np.where(
            n_k > 0, cum_lab[:, k - 1] / np.maximum(n_k, 1), prior_arr
        )
    pos = np.clip(sim[:, :K_PRIMARY], 0.0, 1.0)
    num = np.sum(lab[:, :K_PRIMARY] * pos, axis=1)
    den = np.sum(pos, axis=1)
    cols["rag_note_wrate_k20"] = np.where(den > 0, num / np.maximum(den, 1e-12), prior_arr)
    cols["rag_note_sim_top1"] = sim[:, 0]
    n20 = np.maximum(cum_cnt[:, K_PRIMARY - 1], 1)
    cols["rag_note_sim_mean_k20"] = np.sum(sim[:, :K_PRIMARY], axis=1) / n20
    frame = pd.DataFrame({name: cols[name].astype(np.float32) for name in FEATURE_NAMES})
    return frame


def oof_prior(y: np.ndarray, folds: Sequence[Fold]) -> np.ndarray:
    """Per-row fallback rate for TRAIN: the positive rate of the other folds only."""
    prior = np.empty(len(y), dtype=np.float64)
    for idx_in, idx_out in folds:
        prior[idx_out] = y[idx_in].mean()
    return prior


# --------------------------------------------------------------------------------------
# Leakage and sanity assertions
# --------------------------------------------------------------------------------------
def assert_no_same_patient(
    idx: np.ndarray, query_patients: np.ndarray, index_patients: np.ndarray
) -> int:
    """Raise if any neighbor belongs to the query's patient. Returns rows checked."""
    valid = idx >= 0
    same = index_patients[np.clip(idx, 0, None)] == query_patients[:, None]
    if (same & valid).any():
        raise LeakageError(f"{int((same & valid).any(axis=1).sum())} rows retrieved own patient")
    return len(idx)


def assert_folds_patient_disjoint(folds: Sequence[Fold], patients: np.ndarray, n_rows: int) -> None:
    """Held-out sets partition the rows, and no patient spans a fold's train and held-out sides."""
    seen = np.concatenate([out for _, out in folds])
    if len(seen) != n_rows or len(np.unique(seen)) != n_rows:
        raise LeakageError("held-out folds do not partition the TRAIN rows exactly once")
    for number, (idx_in, idx_out) in enumerate(folds):
        if np.intersect1d(patients[idx_in], patients[idx_out]).size:
            raise LeakageError(f"fold {number}: a patient appears in both index and held-out rows")


def assert_neighbors_inside_index(idx: np.ndarray, folds: Sequence[Fold]) -> None:
    """Each OOF neighbor must come from the row's own fold-train side, never from its own fold."""
    for idx_in, idx_out in folds:
        found = np.unique(idx[idx_out])
        found = found[found >= 0]
        if np.setdiff1d(found, idx_in).size:
            raise LeakageError("an OOF neighbor lies outside the fold's index side")


def assert_same_k_grid(grids: Dict[str, Sequence[int]]) -> None:
    reference = tuple(next(iter(grids.values())))
    for split, grid in grids.items():
        if tuple(grid) != reference:
            raise LeakageError(f"k grid differs for {split}: {tuple(grid)} vs {reference}")


def assert_features_valid(frame: pd.DataFrame, n_neighbors_full: bool, k_grid=K_GRID) -> None:
    """No NaN/inf, rates in [0, 1] and, when every row has k neighbors, on the 1/k grid."""
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("feature frame contains NaN or infinite values")
    for k in k_grid:
        rate = frame[f"rag_note_rate_k{k}"].to_numpy(dtype=np.float64)
        if rate.min() < 0 or rate.max() > 1:
            raise AssertionError(f"rate_k{k} outside [0, 1]")
        if n_neighbors_full and not np.allclose(rate * k, np.round(rate * k), atol=1e-3):
            raise AssertionError(f"rate_k{k} values are not multiples of 1/{k}")


def assert_full_neighbors(idx: np.ndarray, k: int) -> None:
    if idx.shape[1] != k or (idx < 0).any():
        raise AssertionError("some rows have fewer than k neighbors")


# --------------------------------------------------------------------------------------
# Independent re-derivation and negative control
# --------------------------------------------------------------------------------------
def brute_force_row(
    query: np.ndarray,
    query_patient: Any,
    index_vecs: np.ndarray,
    index_patients: np.ndarray,
    index_labels: np.ndarray,
    prior: float,
    k_grid: Sequence[int] = K_GRID,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Slow, loop-based re-derivation of one row's neighbors and features (float64)."""
    sims = index_vecs.astype(np.float64) @ query.astype(np.float64)
    eligible = np.flatnonzero(index_patients != query_patient)
    rounded = np.round(sims[eligible], SIM_DECIMALS)
    order = eligible[np.lexsort((eligible, -rounded))][: max(k_grid)]
    labels = index_labels[order].astype(np.float64)
    feats: Dict[str, float] = {}
    for k in k_grid:
        feats[f"rag_note_rate_k{k}"] = float(labels[:k].mean()) if len(order) else prior
    weights = np.clip(sims[order][:K_PRIMARY], 0.0, 1.0)
    if weights.sum() > 0:
        feats["rag_note_wrate_k20"] = float((labels[:K_PRIMARY] * weights).sum() / weights.sum())
    else:
        feats["rag_note_wrate_k20"] = prior
    feats["rag_note_sim_top1"] = float(sims[order][0]) if len(order) else 0.0
    feats["rag_note_sim_mean_k20"] = float(sims[order][:K_PRIMARY].mean()) if len(order) else 0.0
    return order, feats


def check_oof_against_brute_force(
    emb: np.ndarray,
    patients: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    oof_idx: np.ndarray,
    oof_frame: pd.DataFrame,
    rows: Sequence[int],
) -> Dict[str, Any]:
    """Re-derive chosen TRAIN rows by brute force; neighbors must match exactly."""
    fold_of = np.empty(len(emb), dtype=np.int64)
    for number, (_, idx_out) in enumerate(folds):
        fold_of[idx_out] = number
    max_feature_gap = 0.0
    for row in rows:
        idx_in = folds[fold_of[row]][0]
        order, feats = brute_force_row(
            emb[row],
            patients[row],
            emb[idx_in],
            patients[idx_in],
            y[idx_in],
            float(y[idx_in].mean()),
        )
        got = oof_idx[row][oof_idx[row] >= 0]
        if not np.array_equal(idx_in[order], got):
            raise AssertionError(f"row {row}: neighbors differ from the brute-force result")
        for name, value in feats.items():
            max_feature_gap = max(max_feature_gap, abs(value - float(oof_frame[name].iloc[row])))
    if max_feature_gap > 1e-5:  # features are stored as float32
        raise AssertionError(f"brute-force feature gap {max_feature_gap:.2e}")
    return {
        "rows_checked": len(rows),
        "neighbors_match_exactly": True,
        "max_feature_gap": max_feature_gap,
    }


def univariate_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def shuffled_label_control(
    val_idx: np.ndarray,
    val_sims: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int = SEED,
) -> float:
    """Validation AUC of rate_k20 built from SHUFFLED train labels; should be about 0.5."""
    shuffled = np.random.default_rng(seed).permutation(y_train)
    frame = features_from_neighbors(val_idx, val_sims, shuffled, float(shuffled.mean()))
    return univariate_auc(y_val, frame[PRIMARY_FEATURE].to_numpy())


# --------------------------------------------------------------------------------------
# SQLite vector store (sqlite-vec) used to cross-check exact search
# --------------------------------------------------------------------------------------
class SqliteStoreUnavailable(RuntimeError):
    """sqlite-vec cannot be loaded in this Python build."""


def open_sqlite_store(path: Path) -> sqlite3.Connection:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise SqliteStoreUnavailable("package sqlite-vec is not installed") from exc
    conn = sqlite3.connect(str(path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        raise SqliteStoreUnavailable(f"cannot load the sqlite-vec extension: {exc}") from exc
    return conn


def build_sqlite_store(
    path: Path,
    emb: np.ndarray,
    encounter_ids: np.ndarray,
    patients: np.ndarray,
    labels: np.ndarray,
    batch: int = 5000,
) -> sqlite3.Connection:
    """Create a persistent TRAIN store: cosine vec0 table plus an encounter id table."""
    if path.exists():
        path.unlink()
    conn = open_sqlite_store(path)
    dim = emb.shape[1]
    conn.execute(
        f"CREATE VIRTUAL TABLE notes USING vec0("
        f"embedding float[{dim}] distance_metric=cosine, patient_nbr integer, label integer)"
    )
    conn.execute("CREATE TABLE encounters(row_id INTEGER PRIMARY KEY, encounter_id INTEGER)")
    for lo in range(0, len(emb), batch):
        hi = min(lo + batch, len(emb))
        conn.executemany(
            "INSERT INTO notes(rowid, embedding, patient_nbr, label) VALUES (?, ?, ?, ?)",
            [(int(i), emb[i].tobytes(), int(patients[i]), int(labels[i])) for i in range(lo, hi)],
        )
        conn.executemany(
            "INSERT INTO encounters VALUES (?, ?)",
            [(int(i), int(encounter_ids[i])) for i in range(lo, hi)],
        )
    conn.commit()
    return conn


def sqlite_query(conn: sqlite3.Connection, vec: np.ndarray, patient: int, k: int) -> List[int]:
    """Row positions of the k nearest stored notes from other patients."""
    rows = conn.execute(
        "SELECT rowid FROM notes WHERE embedding MATCH ? AND k = ? AND patient_nbr != ?",
        (vec.astype(np.float32).tobytes(), int(k), int(patient)),
    ).fetchall()
    return [int(r[0]) for r in rows]


def sqlite_recall_at_k(
    conn: sqlite3.Connection,
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    query_patients: np.ndarray,
    exact_idx: np.ndarray,
    exact_sims: np.ndarray,
    n_queries: int = 500,
    k: int = K_PRIMARY,
    seed: int = SEED,
    tie_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Compare the SQLite store with exact search on random validation queries.

    `id_recall` counts shared neighbor ids. `tie_aware_recall` also accepts a neighbor whose
    exact similarity reaches the exact k-th similarity, because identical notes tie.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(query_emb), size=min(n_queries, len(query_emb)), replace=False)
    id_hits, tie_hits = [], []
    for q in chosen:
        found = sqlite_query(conn, query_emb[q], query_patients[q], k)
        truth = set(exact_idx[q, :k].tolist())
        id_hits.append(len(truth & set(found)) / k)
        sims_found = train_emb[found].astype(np.float64) @ query_emb[q].astype(np.float64)
        good = int(np.sum(sims_found >= exact_sims[q, k - 1] - tie_tolerance))
        tie_hits.append(min(good, k) / k)
    return {
        "queries": int(len(chosen)),
        "k": k,
        "id_recall_at_k": float(np.mean(id_hits)),
        "tie_aware_recall_at_k": float(np.mean(tie_hits)),
        "seed": seed,
    }


# --------------------------------------------------------------------------------------
# Convenience: full pipeline for one retriever
# --------------------------------------------------------------------------------------
@dataclass
class TrainNeighbors:
    folds: List[Fold]
    idx: np.ndarray
    sims: np.ndarray
    frame: pd.DataFrame


def build_train_features(
    emb: np.ndarray, patients: np.ndarray, y: np.ndarray, k_max: Optional[int] = None
) -> TrainNeighbors:
    """OOF neighbors and features for TRAIN rows."""
    folds = make_folds(y, patients)
    assert_folds_patient_disjoint(folds, patients, len(y))
    idx, sims = oof_neighbors(emb, patients, folds, k_max or max(K_GRID))
    assert_neighbors_inside_index(idx, folds)
    frame = features_from_neighbors(idx, sims, y, oof_prior(y, folds))
    return TrainNeighbors(folds, idx, sims, frame)


def build_query_features(
    emb_q: np.ndarray,
    patients_q: np.ndarray,
    train_emb: np.ndarray,
    train_patients: np.ndarray,
    y_train: np.ndarray,
    k_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Neighbors and features for validation or test rows against the full TRAIN index."""
    idx, sims = exact_search(emb_q, train_emb, patients_q, train_patients, k_max or max(K_GRID))
    frame = features_from_neighbors(idx, sims, y_train, float(y_train.mean()))
    return idx, sims, frame


def null_auc_tolerance(y: np.ndarray, z: float = 4.0) -> float:
    """z standard errors of the AUC under the null (no signal), from the class counts."""
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    return z * float(np.sqrt((n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg)))
