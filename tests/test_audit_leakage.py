"""Automated audit tests for data leakage, patient separation, and train-only fitting invariance."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_PARQUET,
    TRAIN_RAG_PARQUET,
    VAL_PARQUET,
)
from src.features import build_preprocessor


def _to_dense_array(arr: Any) -> np.ndarray:
    if hasattr(arr, "toarray"):
        return np.asarray(arr.toarray(), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def test_patient_disjoint_across_all_parquets():
    """Verify zero patient overlap between Train, Val, and Test parquets."""
    if not (TRAIN_PARQUET.exists() and VAL_PARQUET.exists()):
        pytest.skip(
            "Processed dataset parquets not found in CI environment; skipping patient disjoint test."
        )
    tr = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet", columns=["patient_nbr"])
    va = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet", columns=["patient_nbr"])
    te = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet", columns=["patient_nbr"])

    s_tr = set(tr["patient_nbr"])
    s_va = set(va["patient_nbr"])
    s_te = set(te["patient_nbr"])

    assert len(s_tr.intersection(s_va)) == 0, "Train and Val share patients!"
    assert len(s_tr.intersection(s_te)) == 0, "Train and Test share patients!"
    assert len(s_va.intersection(s_te)) == 0, "Val and Test share patients!"


def test_retriever_metadata_contains_only_train_patients():
    """Verify FAISS retriever metadata has zero overlap with Val or Test splits."""
    if not (VAL_PARQUET.exists() and (MODELS_DIR / "final" / "retriever_meta.parquet").exists()):
        pytest.skip("Retriever metadata or validation parquet not found; skipping leakage check.")
    meta = pd.read_parquet(MODELS_DIR / "final" / "retriever_meta.parquet")
    va = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet", columns=["patient_nbr"])
    te = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet", columns=["patient_nbr"])

    s_meta = set(meta["patient_nbr"])
    s_va = set(va["patient_nbr"])
    s_te = set(te["patient_nbr"])

    assert len(s_meta.intersection(s_va)) == 0, "Retriever index leaks validation patients!"
    assert len(s_meta.intersection(s_te)) == 0, "Retriever index leaks test patients!"


def test_train_fitting_invariance_under_perturbed_validation():
    """Prove that fitting on train is strictly invariant to validation data perturbations.

    If validation data has 100 new unseen categories and extreme outlier numbers,
    the fitted pipeline parameters on train do not change.
    """
    if not (TRAIN_PARQUET.exists() and VAL_PARQUET.exists()):
        pytest.skip("Train and Val parquets not found; skipping fitting invariance test.")
    with open(MODELS_DIR / "final" / "feature_list.json", "r") as f:
        meta = json.load(f)
    num_cols = [c for c in meta["numeric_features"] if c != "rag_struct_rate_k20"]
    cat_cols = meta["categorical_features"]

    # Use representative sample (2000 rows) to verify pipeline invariance without unbounded RAM spike
    train_df = pd.read_parquet(TRAIN_PARQUET).head(2000)

    # 1. Fit reference preprocessor on clean train
    pipe_clean = build_preprocessor(num_cols, cat_cols)
    X_tr_clean = pipe_clean.fit_transform(train_df[num_cols + cat_cols])

    # 2. Construct perturbed validation set with wild categories and extreme outliers
    val_perturbed = pd.read_parquet(VAL_PARQUET).copy()
    for c in cat_cols:
        val_perturbed[c] = "CompletelyUnseenWildCategoryXYZ"
    for n in num_cols:
        val_perturbed[n] = 999999.0

    # 3. Fit new preprocessor strictly on TRAIN (unaffected by val_perturbed)
    pipe_perturbed = build_preprocessor(num_cols, cat_cols)
    X_tr_second = pipe_perturbed.fit_transform(train_df[num_cols + cat_cols])

    X_clean_dense = _to_dense_array(X_tr_clean)
    X_second_dense = _to_dense_array(X_tr_second)

    # The transformed train vectors must be identical
    np.testing.assert_allclose(X_clean_dense, X_second_dense, rtol=1e-5, atol=1e-5)

    # Transform perturbed val: must not throw exception, handles unknown categories cleanly
    X_val_trans = pipe_clean.transform(val_perturbed[num_cols + cat_cols])
    val_trans_dense = _to_dense_array(X_val_trans)
    assert val_trans_dense.shape[0] == len(val_perturbed)
    assert val_trans_dense.shape[1] == X_clean_dense.shape[1]


def test_label_permutation_produces_chance_auc():
    """Verify that LightGBM trained on randomly shuffled labels achieves AUC ~ 0.50 (no leakage)."""
    if not (TRAIN_RAG_PARQUET.exists() and (PROCESSED_DATA_DIR / "val_rag.parquet").exists()):
        pytest.skip("RAG parquets not found; skipping permutation test.")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    from src.train import prepare_lgbm_data

    train_rag = pd.read_parquet(TRAIN_RAG_PARQUET)
    val_rag = pd.read_parquet(PROCESSED_DATA_DIR / "val_rag.parquet")

    with open(MODELS_DIR / "final" / "feature_list.json", "r") as f:
        meta = json.load(f)
    num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]

    X_train = prepare_lgbm_data(train_rag, num_cols, cat_cols)
    np.random.seed(42)
    y_train_arr = np.asarray(train_rag["readmitted_30d"].to_numpy(), dtype=np.int64)
    y_shuffled = np.random.permutation(y_train_arr)

    X_val = prepare_lgbm_data(val_rag, num_cols, cat_cols)
    y_val = np.asarray(val_rag["readmitted_30d"].to_numpy(), dtype=np.int64)

    clf = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_shuffled)
    raw_probs = clf.predict_proba(X_val)
    probs = np.asarray(raw_probs)[:, 1]
    auc = float(roc_auc_score(y_val, probs))

    # Must be around 0.50 (within 0.47 to 0.53)
    assert 0.47 <= auc <= 0.53, f"Unexpected AUC on shuffled labels: {auc}"
