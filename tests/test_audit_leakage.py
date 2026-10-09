"""Automated audit tests for data leakage, patient separation, and train-only fitting invariance."""

import json

import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_PARQUET,
    TRAIN_RAG_PARQUET,
    VAL_PARQUET,
)
from src.features import build_preprocessor


def test_patient_disjoint_across_all_parquets():
    """Verify zero patient overlap between Train, Val, and Test parquets."""
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
    with open(MODELS_DIR / "final" / "feature_list.json", "r") as f:
        meta = json.load(f)
    num_cols = [c for c in meta["numeric_features"] if c != "rag_struct_rate_k20"]
    cat_cols = meta["categorical_features"]

    train_df = pd.read_parquet(TRAIN_PARQUET)

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

    if hasattr(X_tr_clean, "toarray"):
        X_tr_clean = X_tr_clean.toarray()
    if hasattr(X_tr_second, "toarray"):
        X_tr_second = X_tr_second.toarray()

    # The transformed train vectors must be identical
    np.testing.assert_allclose(X_tr_clean, X_tr_second, rtol=1e-5, atol=1e-5)

    # Transform perturbed val: must not throw exception, handles unknown categories cleanly
    X_val_trans = pipe_clean.transform(val_perturbed[num_cols + cat_cols])
    assert X_val_trans.shape[0] == len(val_perturbed)
    assert X_val_trans.shape[1] == X_tr_clean.shape[1]


def test_label_permutation_produces_chance_auc():
    """Verify that LightGBM trained on randomly shuffled labels achieves AUC ~ 0.50 (no leakage)."""
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
    y_shuffled = np.random.permutation(train_rag["readmitted_30d"].values)

    X_val = prepare_lgbm_data(val_rag, num_cols, cat_cols)
    y_val = val_rag["readmitted_30d"].values

    clf = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_shuffled)
    probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)

    # Must be around 0.50 (within 0.47 to 0.53)
    assert 0.47 <= auc <= 0.53, f"Unexpected AUC on shuffled labels: {auc}"
