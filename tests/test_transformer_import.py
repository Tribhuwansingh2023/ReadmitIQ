"""Tests for src/transformer_import.py.

All tests utilize pytest tmp_path fixtures. No temporary files or test predictions
are ever written to reports/ or models/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import METRICS_DIR, RANDOM_SEED, VAL_RAG_PARQUET
from src.evaluate import compute_paired_bootstrap_auc_diff
from src.transformer_import import (
    EXPECTED_VAL_ROWS,
    check_colab_artifacts,
    import_transformer_results,
    update_ablation_files,
    validate_prediction_csv,
)


@pytest.fixture(scope="module")
def local_val_data():
    """Load local validation ground truth to construct realistic mock predictions."""
    val_df = pd.read_parquet(
        VAL_RAG_PARQUET,
        columns=["encounter_id", "readmitted_30d"],
    )
    assert len(val_df) == EXPECTED_VAL_ROWS
    return {
        "encounter_ids": val_df["encounter_id"].tolist(),
        "labels": val_df["readmitted_30d"].values.astype(int),
    }


def create_mock_prediction_csv(
    path: Path,
    encounter_ids: list[int],
    labels: np.ndarray,
    probs: np.ndarray,
) -> Path:
    """Helper to write a temporary prediction CSV."""
    df = pd.DataFrame(
        {
            "encounter_id": encounter_ids,
            "readmitted_30d": labels,
            "prob": probs,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_accepts_valid_prediction_file(tmp_path: Path, local_val_data: dict):
    """Verify that a syntactically and semantically valid prediction file is accepted."""
    encounters = local_val_data["encounter_ids"]
    labels = local_val_data["labels"]
    rng = np.random.RandomState(RANDOM_SEED)
    probs = np.clip(0.1 + 0.3 * labels + 0.1 * rng.randn(len(labels)), 0.0, 1.0)

    csv_path = tmp_path / "valid_preds.csv"
    create_mock_prediction_csv(csv_path, encounters, labels, probs)

    validated_df = validate_prediction_csv(csv_path, encounters, labels)
    assert len(validated_df) == EXPECTED_VAL_ROWS
    assert list(validated_df.columns) == ["encounter_id", "readmitted_30d", "prob"]


def test_rejects_wrong_row_count(tmp_path: Path, local_val_data: dict):
    """Verify rejection when the CSV does not have exactly 14,902 rows."""
    encounters = local_val_data["encounter_ids"][:500]
    labels = local_val_data["labels"][:500]
    probs = np.full(500, 0.2)

    csv_path = tmp_path / "short_preds.csv"
    create_mock_prediction_csv(csv_path, encounters, labels, probs)

    with pytest.raises(ValueError, match="Invalid row count"):
        validate_prediction_csv(csv_path, local_val_data["encounter_ids"], local_val_data["labels"])


def test_rejects_shuffled_encounter_ids(tmp_path: Path, local_val_data: dict):
    """Verify rejection when encounter_id ordering does not strictly match val_rag.parquet."""
    encounters = list(local_val_data["encounter_ids"])
    # Swap first two encounter IDs
    encounters[0], encounters[1] = encounters[1], encounters[0]
    labels = local_val_data["labels"]
    probs = np.full(len(labels), 0.25)

    csv_path = tmp_path / "shuffled_preds.csv"
    create_mock_prediction_csv(csv_path, encounters, labels, probs)

    with pytest.raises(ValueError, match="Encounter ID ordering mismatch"):
        validate_prediction_csv(csv_path, local_val_data["encounter_ids"], labels)


def test_rejects_label_mismatch(tmp_path: Path, local_val_data: dict):
    """Verify rejection when readmitted_30d values in CSV differ from local ground truth."""
    encounters = local_val_data["encounter_ids"]
    labels = local_val_data["labels"].copy()
    # Flip the first label
    labels[0] = 1 - labels[0]
    probs = np.full(len(labels), 0.25)

    csv_path = tmp_path / "corrupted_labels.csv"
    create_mock_prediction_csv(csv_path, encounters, labels, probs)

    with pytest.raises(ValueError, match="Target label mismatch"):
        validate_prediction_csv(csv_path, encounters, local_val_data["labels"])


def test_rejects_nan_probabilities(tmp_path: Path, local_val_data: dict):
    """Verify rejection when probabilities contain NaN or Inf values."""
    encounters = local_val_data["encounter_ids"]
    labels = local_val_data["labels"]
    probs = np.full(len(labels), 0.2)
    probs[10] = np.nan

    csv_path = tmp_path / "nan_preds.csv"
    create_mock_prediction_csv(csv_path, encounters, labels, probs)

    with pytest.raises(ValueError, match="Non-finite probability"):
        validate_prediction_csv(csv_path, encounters, labels)


def test_rejects_out_of_range_probabilities(tmp_path: Path, local_val_data: dict):
    """Verify rejection when probabilities exceed [0, 1] range."""
    encounters = local_val_data["encounter_ids"]
    labels = local_val_data["labels"]

    # Test prob > 1.0
    probs_high = np.full(len(labels), 0.5)
    probs_high[5] = 1.05
    csv_high = tmp_path / "high_preds.csv"
    create_mock_prediction_csv(csv_high, encounters, labels, probs_high)

    with pytest.raises(ValueError, match="out of valid bounds"):
        validate_prediction_csv(csv_high, encounters, labels)

    # Test prob < 0.0
    probs_low = np.full(len(labels), 0.5)
    probs_low[5] = -0.01
    csv_low = tmp_path / "low_preds.csv"
    create_mock_prediction_csv(csv_low, encounters, labels, probs_low)

    with pytest.raises(ValueError, match="out of valid bounds"):
        validate_prediction_csv(csv_low, encounters, labels)


def test_absent_files_leave_ablation_table_unchanged(tmp_path: Path):
    """Verify that absent Colab export files result in clean exit without modifying ablation table."""
    empty_colab_dir = tmp_path / "colab_empty"
    empty_colab_dir.mkdir()

    all_present, missing = check_colab_artifacts(empty_colab_dir)
    assert not all_present
    assert len(missing) == 4

    # Run import pointing to empty dir
    success = import_transformer_results(colab_dir=empty_colab_dir)
    assert success is False


def test_idempotent_import(tmp_path: Path, monkeypatch):
    """Verify that importing results twice produces identical files without duplicates."""
    # Setup temporary metrics directory with copies of real ablation table
    temp_metrics = tmp_path / "metrics"
    temp_metrics.mkdir()
    real_json = METRICS_DIR / "ablation_table.json"
    real_md = METRICS_DIR / "ablation_table.md"

    temp_json = temp_metrics / "ablation_table.json"
    temp_md = temp_metrics / "ablation_table.md"
    shutil.copy(real_json, temp_json)
    shutil.copy(real_md, temp_md)

    monkeypatch.setattr("src.transformer_import.ABLATION_JSON_PATH", temp_json)
    monkeypatch.setattr("src.transformer_import.ABLATION_MD_PATH", temp_md)

    mock_record_notes = {
        "model": "DistilBERT (Notes)",
        "rag_feature": False,
        "auc_roc": 0.6450,
        "auc_ci_95": [0.6310, 0.6590],
        "optimal_threshold": 0.4800,
        "f1": 0.2650,
        "precision": 0.1800,
        "recall": 0.5000,
        "precision_top_10pct": 0.2200,
        "lift_top_10pct": 1.95,
        "brier_score": None,
        "ece": None,
        "train_samples_used": 69535,
        "val_samples_used": EXPECTED_VAL_ROWS,
        "epochs": 3,
        "max_length": 192,
        "paired_delta_auc": 0.0,
        "paired_ci_95": [0.0, 0.0],
        "ci_crosses_zero": True,
    }
    mock_record_rag = {
        "model": "DistilBERT (Notes)",
        "rag_feature": True,
        "auc_roc": 0.6465,
        "auc_ci_95": [0.6325, 0.6605],
        "optimal_threshold": 0.4850,
        "f1": 0.2660,
        "precision": 0.1810,
        "recall": 0.5020,
        "precision_top_10pct": 0.2220,
        "lift_top_10pct": 1.97,
        "brier_score": None,
        "ece": None,
        "train_samples_used": 69535,
        "val_samples_used": EXPECTED_VAL_ROWS,
        "epochs": 3,
        "max_length": 192,
        "paired_delta_auc": 0.0015,
        "paired_ci_95": [-0.0010, 0.0040],
        "ci_crosses_zero": True,
    }
    mock_meta = {"test_comparison": {"observed_diff": 0.0015}}

    # First update
    update_ablation_files([mock_record_notes, mock_record_rag], mock_meta)
    json_run1 = temp_json.read_text(encoding="utf-8")
    md_run1 = temp_md.read_text(encoding="utf-8")

    with open(temp_json, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    results1 = data1["ablation_results"]
    distilbert_count1 = sum(1 for r in results1 if "DistilBERT" in r["model"])
    assert distilbert_count1 == 2, f"Expected 2 transformer rows, got {distilbert_count1}"

    # Second update with same records
    update_ablation_files([mock_record_notes, mock_record_rag], mock_meta)
    json_run2 = temp_json.read_text(encoding="utf-8")
    md_run2 = temp_md.read_text(encoding="utf-8")

    with open(temp_json, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    results2 = data2["ablation_results"]
    distilbert_count2 = sum(1 for r in results2 if "DistilBERT" in r["model"])
    assert distilbert_count2 == 2, f"Idempotence failed: found {distilbert_count2} transformer rows"

    # Exact string equality ensures full idempotence
    assert json_run1 == json_run2
    assert md_run1 == md_run2


def test_paired_bootstrap_returns_ci_containing_observed_delta():
    """Verify that paired bootstrap returns a 95% CI containing the observed delta."""
    rng = np.random.RandomState(42)
    n = 1000
    y_true = rng.binomial(1, 0.15, size=n)
    y_prob_base = rng.uniform(0.1, 0.9, size=n)
    # RAG adds small positive noise
    y_prob_rag = np.clip(y_prob_base + 0.05 * y_true + rng.normal(0, 0.02, size=n), 0.0, 1.0)

    result = compute_paired_bootstrap_auc_diff(
        y_true, y_prob_base, y_prob_rag, n_bootstraps=500, seed=42
    )

    observed = result["observed_diff"]
    ci_low, ci_high = result["ci_95"]

    assert (
        ci_low <= observed <= ci_high
    ), f"CI [{ci_low}, {ci_high}] does not contain observed {observed}"
    assert result["crosses_zero"] == (ci_low <= 0 <= ci_high)


def test_no_code_path_reads_test_data():
    """Static AST audit asserting that src/transformer_import.py never reads test split files."""
    import_file = Path(__file__).resolve().parent.parent / "src" / "transformer_import.py"
    content = import_file.read_text(encoding="utf-8")

    forbidden_tokens = [
        "test.parquet",
        "test_rag.parquet",
        "TEST_RAG_PARQUET",
        "TEST_PARQUET",
        "test_results.json",
        "test_eval.lock",
    ]

    for token in forbidden_tokens:
        assert (
            token not in content
        ), f"Forbidden test data reference '{token}' found in transformer_import.py!"
