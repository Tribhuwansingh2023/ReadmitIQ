from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import (
    FEATURE_LIST_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)
from src.train import prepare_lgbm_data


def freeze_model_selection():
    """Select winning validation model, calculate thresholds, freeze selection, and save to models/final/."""
    print("=== STAGE 1: Freezing Model Selection ===")

    final_dir = MODELS_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    with open(FEATURE_LIST_PATH, "r") as f:
        f_list = json.load(f)

    base_all = f_list["base_features"]
    rag_col = "rag_struct_rate_k20"
    rag_num_cols = sorted([c for c in base_all if pd.api.types.is_numeric_dtype(pd.Series([0]))] + [rag_col])

    # Re-verify numeric vs categorical columns from train parquet schema
    train_sample = pd.read_parquet(TRAIN_RAG_PARQUET, columns=base_all + [rag_col]).head(10)
    cat_cols = [c for c in base_all if not pd.api.types.is_numeric_dtype(train_sample[c])]
    num_cols = [c for c in base_all if pd.api.types.is_numeric_dtype(train_sample[c])]
    rag_num_cols = sorted(num_cols + [rag_col])

    feature_cols = rag_num_cols + cat_cols

    val_df = pd.read_parquet(VAL_RAG_PARQUET, columns=feature_cols + ["readmitted_30d"])
    y_val = val_df["readmitted_30d"].values.astype(int)

    # 2. Load trained LightGBM RAG model and Calibrator
    lgbm_model = joblib.load(MODELS_DIR / "lgbm_rag.joblib")
    calibrator = joblib.load(MODELS_DIR / "calibrator.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor_rag.joblib")
    svd = joblib.load(MODELS_DIR / "structured_svd.joblib")

    # 3. Predict on Validation and Calibrate
    X_val_lgbm = prepare_lgbm_data(val_df, rag_num_cols, cat_cols)
    raw_val_probs = lgbm_model.predict_proba(X_val_lgbm)[:, 1]
    cal_val_probs = calibrator.transform(raw_val_probs)

    # 4. Compute optimal decision threshold for F1 on validation
    precs, recs, ths = precision_recall_curve(y_val, cal_val_probs)
    denom = precs + recs
    f1s = np.divide(2 * precs * recs, denom, out=np.zeros_like(precs), where=(denom > 0))
    best_idx = int(np.argmax(f1s))
    best_th = float(ths[best_idx]) if best_idx < len(ths) else 0.5

    # 5. Compute Capacity Thresholds on Validation (Percentiles)
    # top 5% means top 5% highest scores -> 95th percentile
    th_top_5 = float(np.percentile(cal_val_probs, 95.0))
    th_top_10 = float(np.percentile(cal_val_probs, 90.0))
    th_top_20 = float(np.percentile(cal_val_probs, 80.0))

    print(f"Validation Thresholds Calculated:")
    print(f"  Optimal F1 Decision Threshold: {best_th:.4f}")
    print(f"  Capacity Threshold (Top 5%):   {th_top_5:.4f}")
    print(f"  Capacity Threshold (Top 10%):  {th_top_10:.4f}")
    print(f"  Capacity Threshold (Top 20%):  {th_top_20:.4f}")

    # 6. Retrieve git commit hash
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )

    now_utc = datetime.now(timezone.utc).isoformat()

    frozen_metadata = {
        "frozen_timestamp": now_utc,
        "git_commit_hash": git_hash,
        "model_name": "LightGBM + Structured RAG",
        "model_family": "LightGBMClassifier",
        "feature_set": "rag_features",
        "num_features": len(feature_cols),
        "primary_rag_feature": rag_col,
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "scale_pos_weight": 7.7,
            "random_state": RANDOM_SEED,
        },
        "retriever": {
            "type": "StructuredRAGEngine",
            "index_type": "IndexFlatIP",
            "projection": "TruncatedSVD (64 components)",
            "normalization": "L2",
            "similarity_metric": "Cosine Similarity (Inner Product)",
            "k_neighbors": 20,
            "k_search": 50,
            "train_encounters_indexed": 69535,
        },
        "calibrator": {
            "method": "IsotonicRegression",
            "fit_split": "TRAIN_5_FOLD_OOF",
            "out_of_bounds": "clip",
            "val_brier_score": 0.0966,
            "val_ece": 0.0024,
        },
        "thresholds": {
            "optimal_f1_decision_threshold": round(best_th, 4),
            "capacity_top_5pct_threshold": round(th_top_5, 4),
            "capacity_top_10pct_threshold": round(th_top_10, 4),
            "capacity_top_20pct_threshold": round(th_top_20, 4),
        },
        "validation_performance": {
            "auc_roc": 0.6731,
            "auc_ci_95": [0.6602, 0.6863],
            "precision_top_10pct": 0.2530,
            "lift_top_10pct": 2.22,
            "f1": 0.2822,
        },
        "random_seed": RANDOM_SEED,
        "status": "FROZEN",
        "tuning_prohibited": True,
    }

    frozen_json_path = MODELS_DIR / "FROZEN_SELECTION.json"
    with open(frozen_json_path, "w", encoding="utf-8") as f:
        json.dump(frozen_metadata, f, indent=2)
    print(f"Created {frozen_json_path}")

    # 7. Persist final models and artifacts in models/final/
    joblib.dump(lgbm_model, final_dir / "model.joblib")
    joblib.dump(preprocessor, final_dir / "preprocessor.joblib")
    joblib.dump(calibrator, final_dir / "calibrator.joblib")
    joblib.dump(svd, final_dir / "structured_svd.joblib")

    # Copy FAISS index and metadata
    shutil.copy2(MODELS_DIR / "rag_struct_faiss.index", final_dir / "retriever_faiss.index")
    shutil.copy2(MODELS_DIR / "rag_train_meta.parquet", final_dir / "retriever_meta.parquet")

    # Copy feature list
    with open(final_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "numeric_features": rag_num_cols,
                "categorical_features": cat_cols,
                "all_features": feature_cols,
                "target": "readmitted_30d",
            },
            f,
            indent=2,
        )

    print(f"All final serving artifacts successfully saved to {final_dir}")
    print("STAGE 1 COMPLETE: Model is strictly frozen.")


if __name__ == "__main__":
    freeze_model_selection()
