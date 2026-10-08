import json
import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_LIST_PATH,
    FEATURE_METADATA_PATH,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.features import build_preprocessor


@pytest.fixture(scope="module")
def data_splits():
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    with open(FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    return train_df, val_df, test_df, meta


def test_no_nulls_in_features(data_splits):
    """Test that all engineered feature columns have zero missing values across all splits."""
    train_df, val_df, test_df, meta = data_splits
    base_numeric = [c for c in meta.get("base_numeric_features", meta["numeric_features"]) if c != "rag_readmit_rate_k20"]
    feature_cols = base_numeric + meta["categorical_features"]

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        null_counts = df[feature_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert (
            len(cols_with_nulls) == 0
        ), f"Missing values found in {name} split: {cols_with_nulls.to_dict()}"


def test_target_binary(data_splits):
    """Test that readmitted_30d is strictly a binary {0, 1} integer."""
    train_df, val_df, test_df, _ = data_splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        unique_vals = set(df["readmitted_30d"].unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary target in {name}: {unique_vals}"


def test_id_columns_not_in_features(data_splits):
    """Test that encounter_id and patient_nbr are excluded from model features."""
    with open(FEATURE_LIST_PATH, "r") as f:
        f_list = json.load(f)

    all_features = set(f_list.get("all_features_with_rag", f_list["base_features"]))
    assert "encounter_id" not in all_features, "encounter_id leaked into features!"
    assert "patient_nbr" not in all_features, "patient_nbr leaked into features!"
    assert "readmitted" not in all_features, "readmitted label leaked into features!"
    assert "readmitted_30d" not in all_features, "target leaked into features!"


def test_preprocessor_pipeline_fit_transform(data_splits):
    """Test that the sklearn Preprocessor fits cleanly on train and transforms val without error."""
    train_df, val_df, _, meta = data_splits
    num_cols = [c for c in meta["numeric_features"] if c in train_df.columns]
    cat_cols = meta["categorical_features"]
    feature_cols = num_cols + cat_cols

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(train_df[feature_cols])
    X_val_trans = preprocessor.transform(val_df[feature_cols])

    assert not np.isnan(X_train_trans).any(), "NaN detected in transformed train features"
    assert not np.isnan(X_val_trans).any(), "NaN detected in transformed val features"
    assert X_train_trans.shape[1] == X_val_trans.shape[1], "Feature dimension mismatch between train and val"
    assert X_train_trans.shape[1] < 150, f"Transformed column count {X_train_trans.shape[1]} exceeds target of 150!"


def test_rare_category_collapsing(data_splits):
    """Test that rare categories in medical_specialty and payer_code are properly collapsed."""
    train_df, val_df, _, meta = data_splits
    num_cols = [c for c in meta["numeric_features"] if c in train_df.columns]
    cat_cols = meta["categorical_features"]
    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(train_df[num_cols + cat_cols])

    # Check grouper inside categorical transformer
    cat_pipe = preprocessor.named_transformers_["cat"]
    grouper = cat_pipe.named_steps["grouper"]
    assert len(grouper.top_specialties_) <= 10, f"Expected at most 10 top specialties, got {len(grouper.top_specialties_)}"
    assert len(grouper.frequent_payers_) > 0, "No frequent payers learned"

    # Verify transformed column count is well under 150
    feature_names = preprocessor.get_feature_names_out()
    assert len(feature_names) == 141, f"Expected exactly 141 columns, got {len(feature_names)}"
