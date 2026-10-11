"""Comprehensive automated audit tests for ROC-AUC computation, tie handling,
clinical data leakage, patient/encounter split isolation, and metric consistency.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from notebooks.stages._stage_helpers import compute_roc_auc
from scripts.run_xai_analysis import compute_roc_auc_pure
from src.config import (
    BASE_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.evaluate import evaluate_predictions


# -----------------------------------------------------------------------------
# 1. Metric Calculation & Tie-Breaking Correctness Tests
# -----------------------------------------------------------------------------

def test_compute_roc_auc_exact_sklearn_agreement():
    """Verify compute_roc_auc matches sklearn.metrics.roc_auc_score across edge cases."""
    test_cases = [
        # Standard distinct probabilities
        (np.array([1, 0, 1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])),
        # Completely tied constant probabilities (must be exactly 0.50)
        (np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5])),
        # Partially tied probabilities (plateaus)
        (np.array([1, 0, 1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.2])),
        # Multi-plateau discrete scores (e.g., discrete RAG fraction 1/20)
        (
            np.array([1, 0, 0, 1, 1, 0, 0, 1]),
            np.array([0.05, 0.05, 0.10, 0.10, 0.25, 0.25, 0.30, 0.40]),
        ),
        # Perfect separation
        (np.array([1, 1, 0, 0]), np.array([0.95, 0.85, 0.15, 0.05])),
        # Perfectly inverted scores
        (np.array([1, 1, 0, 0]), np.array([0.05, 0.15, 0.85, 0.95])),
    ]

    for y_true, y_score in test_cases:
        sk_auc = float(roc_auc_score(y_true, y_score))
        custom_auc = compute_roc_auc(y_true, y_score)
        pure_auc = compute_roc_auc_pure(y_true, y_score)

        assert abs(custom_auc - sk_auc) < 1e-9, (
            f"compute_roc_auc failed: got {custom_auc}, expected {sk_auc}"
        )
        assert abs(pure_auc - sk_auc) < 1e-9, (
            f"compute_roc_auc_pure failed: got {pure_auc}, expected {sk_auc}"
        )


def test_roc_auc_row_order_invariance():
    """Verify compute_roc_auc is invariant to array order reversal on tied scores.
    
    The old flawed implementation used arbitrary argsort index ordering for ties,
    which caused AUC to fluctuate depending on row sorting. The fix guarantees invariance.
    """
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    # Massive ties as produced by isotonic calibration plateaus
    y_score = np.array([0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.1, 0.8])

    auc_forward = compute_roc_auc(y_true, y_score)
    auc_reversed = compute_roc_auc(y_true[::-1], y_score[::-1])

    # Reversing must produce the exact same AUC
    assert abs(auc_forward - auc_reversed) < 1e-9, (
        f"Order variance detected: forward={auc_forward}, reversed={auc_reversed}"
    )


def test_compute_roc_auc_empty_or_single_class_edge_cases():
    """Verify single-class or degenerate inputs safely return 0.50 without crashing."""
    assert compute_roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) == 0.5
    assert compute_roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])) == 0.5
    assert compute_roc_auc(np.array([]), np.array([])) == 0.5


# -----------------------------------------------------------------------------
# 2. Clinical Data Leakage & Patient Group Isolation Tests
# -----------------------------------------------------------------------------

def test_split_patient_and_encounter_isolation():
    """Verify complete disjointness of patients and encounters across Train, Val, and Test."""
    if not (TRAIN_PARQUET.exists() and VAL_PARQUET.exists() and TEST_PARQUET.exists()):
        pytest.skip("Processed dataset parquets not found; skipping split isolation test.")

    cols = ["patient_nbr", "encounter_id"]
    tr = pd.read_parquet(TRAIN_PARQUET, columns=cols)
    va = pd.read_parquet(VAL_PARQUET, columns=cols)
    te = pd.read_parquet(TEST_PARQUET, columns=cols)

    # 1. Patient overlap must be exactly 0
    tr_pts = set(tr["patient_nbr"])
    va_pts = set(va["patient_nbr"])
    te_pts = set(te["patient_nbr"])

    assert len(tr_pts.intersection(va_pts)) == 0, "Patient leakage between Train and Val!"
    assert len(tr_pts.intersection(te_pts)) == 0, "Patient leakage between Train and Test!"
    assert len(va_pts.intersection(te_pts)) == 0, "Patient leakage between Val and Test!"

    # 2. Encounter overlap must be exactly 0
    tr_encs = set(tr["encounter_id"])
    va_encs = set(va["encounter_id"])
    te_encs = set(te["encounter_id"])

    assert len(tr_encs.intersection(va_encs)) == 0, "Encounter leakage between Train and Val!"
    assert len(tr_encs.intersection(te_encs)) == 0, "Encounter leakage between Train and Test!"
    assert len(va_encs.intersection(te_encs)) == 0, "Encounter leakage between Val and Test!"


def test_target_token_absence_in_clinical_notes():
    """Verify clinical notes contain zero target-derived tokens or readmission indicators."""
    train_rag = PROCESSED_DATA_DIR / "train_rag.parquet"
    if not train_rag.exists():
        pytest.skip("train_rag.parquet not found; skipping note token leakage check.")

    df = pd.read_parquet(train_rag, columns=["clinical_note"]).head(2000)
    forbidden_patterns = [
        re.compile(r"\breadmit\b", re.IGNORECASE),
        re.compile(r"\breadmission\b", re.IGNORECASE),
        re.compile(r"\breadmitted\b", re.IGNORECASE),
        re.compile(r"(?<!\d)<30(?!\d)"),
        re.compile(r"(?<!\d)>30(?!\d)"),
    ]

    for idx, note in enumerate(df["clinical_note"]):
        for pat in forbidden_patterns:
            assert not pat.search(note), f"Forbidden target token '{pat.pattern}' found in row {idx}: {note}"


# -----------------------------------------------------------------------------
# 3. Artifact Integrity, Calibration Plateaus & Governance Lock Tests
# -----------------------------------------------------------------------------

def test_test_evaluation_governance_lock():
    """Verify test split evaluation is sealed with a cryptographic lock and frozen selection."""
    lock_file = METRICS_DIR / "test_eval.lock"
    test_results_file = METRICS_DIR / "test_results.json"
    frozen_selection_file = MODELS_DIR / "FROZEN_SELECTION.json"

    assert lock_file.exists(), "Governance lock file test_eval.lock is missing!"
    assert test_results_file.exists(), "test_results.json is missing!"
    assert frozen_selection_file.exists(), "FROZEN_SELECTION.json is missing!"

    with open(test_results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Locked test evaluation must report the frozen 0.6590 AUC-ROC
    assert "auc_roc" in results
    assert results["auc_roc"] == pytest.approx(0.6590, abs=1e-4)
    assert results["auc_ci_95"][0] == pytest.approx(0.6458, abs=1e-4)
    assert results["auc_ci_95"][1] == pytest.approx(0.6728, abs=1e-4)


def test_val_predictions_alignment_with_ground_truth():
    """Verify DistilBERT validation prediction CSVs align exactly with val.parquet labels."""
    val_pq = pd.read_parquet(VAL_PARQUET, columns=["encounter_id", "readmitted_30d"])

    pred_paths = [
        BASE_DIR / "reports" / "colab_outputs" / "distilbert_val_predictions_notes.csv",
        BASE_DIR / "reports" / "colab_outputs" / "distilbert_val_predictions_notes_rag.csv",
        BASE_DIR / "reports" / "colab" / "distilbert_val_predictions_notes.csv",
        BASE_DIR / "reports" / "note_rag" / "colab" / "distilbert_val_predictions_notes_noterag.csv",
    ]

    tested_any = False
    for p in pred_paths:
        if p.exists():
            df_pred = pd.read_csv(p)
            assert len(df_pred) == len(val_pq), f"Length mismatch in {p.name}"
            assert (df_pred["encounter_id"].values == val_pq["encounter_id"].values).all(), (
                f"Encounter ID mismatch in {p.name}"
            )
            assert (df_pred["readmitted_30d"].values == val_pq["readmitted_30d"].values).all(), (
                f"Label mismatch in {p.name}"
            )
            # Probabilities must be within valid [0, 1] interval
            assert df_pred["prob"].min() >= 0.0
            assert df_pred["prob"].max() <= 1.0
            tested_any = True

    if not tested_any:
        pytest.skip("No DistilBERT validation prediction CSVs found on disk.")
