import json

import pandas as pd
import pytest

from src.config import (
    FEATURE_METADATA_PATH,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)


@pytest.fixture(scope="module")
def data_splits():
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    with open(FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    return train_df, val_df, test_df, meta


def test_no_patient_overlap(data_splits):
    """Test that patient_nbr is strictly disjoint across train, val, and test splits."""
    train_df, val_df, test_df, _ = data_splits
    train_pts = set(train_df["patient_nbr"])
    val_pts = set(val_df["patient_nbr"])
    test_pts = set(test_df["patient_nbr"])

    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    assert (
        len(train_val_overlap) == 0
    ), f"Found {len(train_val_overlap)} overlapping patients between train and val"
    assert (
        len(train_test_overlap) == 0
    ), f"Found {len(train_test_overlap)} overlapping patients between train and test"
    assert (
        len(val_test_overlap) == 0
    ), f"Found {len(val_test_overlap)} overlapping patients between val and test"


def test_target_rate_per_split(data_splits):
    """Test that readmission rate (~11%) is preserved across train, val, and test splits."""
    train_df, val_df, test_df, _ = data_splits
    train_rate = train_df["readmitted_30d"].mean()
    val_rate = val_df["readmitted_30d"].mean()
    test_rate = test_df["readmitted_30d"].mean()

    assert 0.10 <= train_rate <= 0.13, f"Train target rate abnormal: {train_rate:.4f}"
    assert 0.10 <= val_rate <= 0.13, f"Val target rate abnormal: {val_rate:.4f}"
    assert 0.10 <= test_rate <= 0.13, f"Test target rate abnormal: {test_rate:.4f}"

    assert abs(train_rate - val_rate) < 0.005, "Val rate deviated from train"
    assert abs(train_rate - test_rate) < 0.005, "Test rate deviated from train"
