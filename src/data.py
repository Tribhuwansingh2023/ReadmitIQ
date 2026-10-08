import json
from pathlib import Path
import sys
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    EXCLUDE_DISCHARGE_IDS,
    FEATURE_METADATA_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SPLIT_N_SPLITS,
    TEST_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.features import engineer_features, get_feature_columns


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw UCI diabetic data CSV with proper missing value handling.

    Keeps 'None' as a valid category in test results.
    """
    df = pd.read_csv(
        filepath,
        na_values=["?"],
        keep_default_na=True,
        low_memory=False,
    )
    # Ensure 'None' is maintained as string, not NaN
    for col in ["max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    return df


def split_data(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    n_splits: int = SPLIT_N_SPLITS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform patient-level 70/15/15 split using StratifiedGroupKFold.

    Guarantees no patient_nbr overlap across train, val, and test splits
    while preserving stratified ~11% positive target rate.
    """
    # Clean non-eligible rows before split
    data = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)].copy()
    data = data[data["gender"] != "Unknown/Invalid"].reset_index(drop=True)

    y = (data["readmitted"] == "<30").astype(int).values
    groups = data["patient_nbr"].values

    # Step 1: n_splits fold split -> 70% train (e.g. 7 of 10 folds), 30% temp
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(data), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(data, y, groups)):
        fold_assignments[test_idx] = fold

    train_folds_count = int(round(n_splits * 0.70))
    train_mask = np.isin(fold_assignments, range(0, train_folds_count))
    temp_mask = np.isin(fold_assignments, range(train_folds_count, n_splits))

    train_indices = np.where(train_mask)[0]
    temp_indices = np.where(temp_mask)[0]

    # Step 2: 2-fold split on temp -> 15% val, 15% test
    temp_y = y[temp_indices]
    temp_groups = groups[temp_indices]
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)

    val_indices = None
    test_indices = None
    for fold, (_, sub_idx) in enumerate(sgkf2.split(temp_indices, temp_y, temp_groups)):
        if fold == 0:
            val_indices = temp_indices[sub_idx]
        else:
            test_indices = temp_indices[sub_idx]

    train_df = data.iloc[train_indices].copy().reset_index(drop=True)
    val_df = data.iloc[val_indices].copy().reset_index(drop=True)
    test_df = data.iloc[test_indices].copy().reset_index(drop=True)

    # Verification of zero patient leakage
    train_pts = set(train_df["patient_nbr"])
    val_pts = set(val_df["patient_nbr"])
    test_pts = set(test_df["patient_nbr"])
    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between train and val"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and test"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and test"

    return train_df, val_df, test_df


def process_and_save_data(
    raw_path: Path = RAW_DATA_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Dict:
    """End-to-end data processing: load, split, engineer features, save parquets and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / ".cache_splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train_raw.parquet"
    val_cache = cache_dir / "val_raw.parquet"
    test_cache = cache_dir / "test_raw.parquet"

    if train_cache.exists() and val_cache.exists() and test_cache.exists():
        print("Loading raw patient-level splits from cache...")
        train_raw = pd.read_parquet(train_cache)
        val_raw = pd.read_parquet(val_cache)
        test_raw = pd.read_parquet(test_cache)
    else:
        print("Loading raw dataset...")
        df_raw = load_raw_data(raw_path)

        print("Splitting at patient level (70/15/15)...")
        train_raw, val_raw, test_raw = split_data(df_raw)

        train_raw.to_parquet(train_cache, index=False)
        val_raw.to_parquet(val_cache, index=False)
        test_raw.to_parquet(test_cache, index=False)

    print("Engineering features on TRAIN split...")
    train_feat, capping_thresholds, retained_drugs = engineer_features(
        train_raw, is_train=True
    )

    print("Transforming VAL split with train-fitted statistics...")
    val_feat, _, _ = engineer_features(
        val_raw,
        capping_thresholds=capping_thresholds,
        retained_drugs=retained_drugs,
        is_train=False,
    )

    print("Transforming TEST split with train-fitted statistics...")
    test_feat, _, _ = engineer_features(
        test_raw,
        capping_thresholds=capping_thresholds,
        retained_drugs=retained_drugs,
        is_train=False,
    )

    # Validate feature sets
    numeric_cols, categorical_cols = get_feature_columns(train_feat)

    print("Saving processed parquets...")
    train_feat.to_parquet(output_dir / "train.parquet", index=False)
    val_feat.to_parquet(output_dir / "val.parquet", index=False)
    test_feat.to_parquet(output_dir / "test.parquet", index=False)

    metadata = {
        "num_total_records": len(train_feat) + len(val_feat) + len(test_feat),
        "splits": {
            "train": {
                "rows": len(train_feat),
                "unique_patients": int(train_feat["patient_nbr"].nunique()),
                "positive_count": int(train_feat["readmitted_30d"].sum()),
                "positive_rate": float(train_feat["readmitted_30d"].mean()),
            },
            "val": {
                "rows": len(val_feat),
                "unique_patients": int(val_feat["patient_nbr"].nunique()),
                "positive_count": int(val_feat["readmitted_30d"].sum()),
                "positive_rate": float(val_feat["readmitted_30d"].mean()),
            },
            "test": {
                "rows": len(test_feat),
                "unique_patients": int(test_feat["patient_nbr"].nunique()),
                "positive_count": int(test_feat["readmitted_30d"].sum()),
                "positive_rate": float(test_feat["readmitted_30d"].mean()),
            },
        },
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "retained_drugs": retained_drugs,
        "capping_thresholds_99th": capping_thresholds,
    }

    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Processing complete!")
    return metadata


if __name__ == "__main__":
    process_and_save_data()
