"""Tests for Step 0: Scoring correctness, raw-score cutoffs, and test split audit."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import BASE_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.train import prepare_lgbm_data


@pytest.fixture(scope="module")
def artifacts():
    final_dir = MODELS_DIR / "final"
    model = joblib.load(final_dir / "model.joblib")
    calibrator = joblib.load(final_dir / "calibrator.joblib")
    with open(final_dir / "feature_list.json", "r", encoding="utf-8") as f:
        feat_meta = json.load(f)

    with open(final_dir / "serving_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    with open(BASE_DIR / "reports" / "metrics" / "test_results.json", "r", encoding="utf-8") as f:
        test_results = json.load(f)

    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test_rag.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val_rag.parquet")

    return {
        "model": model,
        "calibrator": calibrator,
        "feat_meta": feat_meta,
        "thresholds": thresholds,
        "test_results": test_results,
        "test_df": test_df,
        "val_df": val_df,
    }


def test_serving_thresholds_match_validation_cutoffs(artifacts):
    """Verify serving_thresholds.json matches validation raw score percentiles."""
    thresholds = artifacts["thresholds"]
    cutoffs = thresholds["raw_score_cutoffs"]

    # Verify cutoffs are strictly descending
    assert cutoffs["top_5pct"] > cutoffs["top_10pct"] > cutoffs["top_20pct"]
    assert cutoffs["top_5pct"] == pytest.approx(0.683933, abs=1e-4)
    assert cutoffs["top_10pct"] == pytest.approx(0.640775, abs=1e-4)
    assert cutoffs["top_20pct"] == pytest.approx(0.579340, abs=1e-4)

    # Verify PSI decile bin edges
    psi_edges = thresholds["psi_reference"]["decile_bin_edges"]
    assert len(psi_edges) == 11
    assert psi_edges[0] == 0.0
    assert psi_edges[-1] == 1.0


def test_isotonic_plateaus_vs_raw_score_cardinality(artifacts):
    """Verify calibrated probability has massive plateaus (~57 distinct values) while raw score has high cardinality (>14,800)."""
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    feat_meta = artifacts["feat_meta"]
    test_df = artifacts["test_df"]

    X_test = prepare_lgbm_data(test_df, feat_meta["numeric_features"], feat_meta["categorical_features"])
    raw_scores = model.predict_proba(X_test)[:, 1]
    cal_probs = calibrator.transform(raw_scores)

    distinct_raw = len(np.unique(raw_scores))
    distinct_cal = len(np.unique(cal_probs))

    # Calibrated probabilities have only ~57 distinct values across 14,903 rows
    assert distinct_cal <= 65
    assert distinct_raw > 14000


def test_test_split_ranking_audit(artifacts):
    """Audit rank-based top-5/10/20% caught counts on held-out test split.

    Examines difference between quicksort on calibrated probabilities (test_results.json)
    and true raw-score ranking with deterministic encounter_id tie-breaking.
    """
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    feat_meta = artifacts["feat_meta"]
    test_df = artifacts["test_df"]
    test_results = artifacts["test_results"]

    X_test = prepare_lgbm_data(test_df, feat_meta["numeric_features"], feat_meta["categorical_features"])
    raw_scores = model.predict_proba(X_test)[:, 1]
    cal_probs = calibrator.transform(raw_scores)

    n_test = len(test_df)
    k_10 = int(n_test * 0.10)  # 1490

    # 1. Calibrated ranking with quicksort yields exactly test_results.json numbers
    # (top_10pct precision 0.2376 * 1490 = 354 caught)
    cal_order = np.argsort(cal_probs)[::-1]
    cal_top_10_caught = int(test_df["readmitted_30d"].iloc[cal_order[:k_10]].sum())
    assert cal_top_10_caught == 354
    assert test_results["capacity_metrics"]["top_10pct"]["precision"] == pytest.approx(354 / k_10, abs=1e-3)

    # 2. Raw score ranking with deterministic encounter_id tie-break:
    # Sort descending by raw_score, ascending by encounter_id
    df_eval = pd.DataFrame({
        "encounter_id": test_df["encounter_id"],
        "raw_score": raw_scores,
        "label": test_df["readmitted_30d"],
    }).sort_values(by=["raw_score", "encounter_id"], ascending=[False, True]).reset_index(drop=True)

    raw_k5_caught = int(df_eval.iloc[:int(n_test * 0.05)]["label"].sum())   # k=745 -> 203 caught
    raw_k10_caught = int(df_eval.iloc[:k_10]["label"].sum())                # k=1490 -> 359 caught
    raw_k20_caught = int(df_eval.iloc[:int(n_test * 0.20)]["label"].sum())  # k=2980 -> 616 caught

    # Exact audited counts
    assert raw_k10_caught == 359, f"Expected 359 caught at 10% by raw score, got {raw_k10_caught}"
    assert raw_k5_caught == 203
    assert raw_k20_caught == 616

    # Audit discrepancy documentation:
    # At top 10%, raw score caught 359 vs 354 in test_results.json (+5 more true readmissions caught
    # because raw score discriminates within the 533-encounter isotonic plateau at cal_prob ~ 0.194).
    diff_at_10 = raw_k10_caught - 354
    assert diff_at_10 == 5
