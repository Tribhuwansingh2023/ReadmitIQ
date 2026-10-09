import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_METADATA_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from src.features import build_preprocessor


def run_baseline_training() -> dict:
    """Train baseline Logistic Regression on TRAIN and evaluate on VAL."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed train and val parquets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)

    with open(FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)

    num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df["readmitted_30d"]

    X_val = val_df[feature_cols]
    y_val = val_df["readmitted_30d"]

    print(f"Features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"Training Logistic Regression baseline (samples: {len(X_train)})...")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluate on VAL split (Touch test set ONLY in Phase 4!)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    auc_roc = float(roc_auc_score(y_val, y_val_proba))

    # Standard threshold 0.5 metrics
    y_val_pred_default = (y_val_proba >= 0.5).astype(int)
    f1_default = float(f1_score(y_val, y_val_pred_default, zero_division=0))
    prec_default = float(precision_score(y_val, y_val_pred_default, zero_division=0))
    rec_default = float(recall_score(y_val, y_val_pred_default, zero_division=0))

    # Tuned threshold for maximum F1
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    denominator = precisions + recalls
    f1_scores = np.divide(
        2 * (precisions * recalls),
        denominator,
        out=np.zeros_like(precisions),
        where=(denominator > 0),
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_val_pred_tuned = (y_val_proba >= best_threshold).astype(int)

    f1_tuned = float(f1_score(y_val, y_val_pred_tuned, zero_division=0))
    prec_tuned = float(precision_score(y_val, y_val_pred_tuned, zero_division=0))
    rec_tuned = float(recall_score(y_val, y_val_pred_tuned, zero_division=0))

    metrics = {
        "model": "Baseline Logistic Regression (Balanced)",
        "eval_split": "validation",
        "auc_roc": round(auc_roc, 4),
        "threshold_0.5": {
            "f1": round(f1_default, 4),
            "precision": round(prec_default, 4),
            "recall": round(rec_default, 4),
        },
        "optimal_threshold": {
            "threshold": round(best_threshold, 4),
            "f1": round(f1_tuned, 4),
            "precision": round(prec_tuned, 4),
            "recall": round(rec_tuned, 4),
        },
    }

    # Save artifacts
    joblib.dump(pipeline, MODELS_DIR / "baseline_lr.joblib")
    with open(REPORTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Validation AUC-ROC: {auc_roc:.4f}")
    print(f"Validation F1 (tuned threshold {best_threshold:.4f}): {f1_tuned:.4f}")
    print(f"Validation Precision: {prec_tuned:.4f}, Recall: {rec_tuned:.4f}")

    return metrics


if __name__ == "__main__":
    run_baseline_training()
