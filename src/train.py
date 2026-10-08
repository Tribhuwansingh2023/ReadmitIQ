import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_LIST_PATH,
    METRICS_DIR,
    MODELS_DIR,
    PREPROCESSOR_BASE_PATH,
    PREPROCESSOR_RAG_PATH,
    RANDOM_SEED,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)
from src.evaluate import evaluate_predictions
from src.features import build_preprocessor


def prepare_lgbm_data(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    """Format dataframe for native LightGBM categorical handling."""
    data = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        data[col] = data[col].astype("category")
    return data


def train_and_evaluate_all_models() -> List[Dict]:
    """Train LR, Random Forest, and LightGBM with and without RAG feature.

    Evaluates on the exact same validation split and saves all models and ablation metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RAG-enhanced train and validation datasets...")
    train_df = pd.read_parquet(TRAIN_RAG_PARQUET)
    val_df = pd.read_parquet(VAL_RAG_PARQUET)

    with open(FEATURE_LIST_PATH, "r") as f:
        f_list = json.load(f)

    base_all = f_list["base_features"]
    rag_col = "rag_readmit_rate_k20"

    cat_cols = [c for c in base_all if not pd.api.types.is_numeric_dtype(train_df[c])]
    base_num_cols = [c for c in base_all if pd.api.types.is_numeric_dtype(train_df[c])]
    rag_num_cols = sorted(base_num_cols + [rag_col])

    y_train = train_df["readmitted_30d"].values
    y_val = val_df["readmitted_30d"].values

    print(f"Cohort sizes: Train={len(train_df)}, Val={len(val_df)}")
    print(f"Features: Base={len(base_all)} ({len(base_num_cols)} num, {len(cat_cols)} cat); With RAG={len(base_all)+1}")

    ablation_records = []

    # -------------------------------------------------------------
    # 1. Logistic Regression: Without RAG
    # -------------------------------------------------------------
    print("\n--- Training Logistic Regression (Without RAG) ---")
    pipe_lr_base = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(base_num_cols, cat_cols)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )
    pipe_lr_base.fit(train_df[base_num_cols + cat_cols], y_train)
    val_prob = pipe_lr_base.predict_proba(val_df[base_num_cols + cat_cols])[:, 1]
    metrics_lr_base = evaluate_predictions(y_val, val_prob)
    metrics_lr_base.update({"model": "Logistic Regression", "rag_feature": False})
    ablation_records.append(metrics_lr_base)
    print(f"Validation AUC: {metrics_lr_base['auc_roc']} (CI: {metrics_lr_base['auc_ci_95']}) | F1: {metrics_lr_base['f1']} | Prec@10%: {metrics_lr_base['precision_top_10pct']} (Lift: {metrics_lr_base['lift_top_10pct']}x)")

    # -------------------------------------------------------------
    # 2. Logistic Regression: With RAG
    # -------------------------------------------------------------
    print("\n--- Training Logistic Regression (With RAG) ---")
    pipe_lr_rag = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(rag_num_cols, cat_cols)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )
    pipe_lr_rag.fit(train_df[rag_num_cols + cat_cols], y_train)
    val_prob = pipe_lr_rag.predict_proba(val_df[rag_num_cols + cat_cols])[:, 1]
    metrics_lr_rag = evaluate_predictions(y_val, val_prob)
    metrics_lr_rag.update({"model": "Logistic Regression", "rag_feature": True})
    ablation_records.append(metrics_lr_rag)
    joblib.dump(pipe_lr_rag, MODELS_DIR / "lr_rag.joblib")
    print(f"Validation AUC: {metrics_lr_rag['auc_roc']} (CI: {metrics_lr_rag['auc_ci_95']}) | F1: {metrics_lr_rag['f1']} | Prec@10%: {metrics_lr_rag['precision_top_10pct']} (Lift: {metrics_lr_rag['lift_top_10pct']}x)")

    # -------------------------------------------------------------
    # 3. Random Forest: Without RAG
    # -------------------------------------------------------------
    print("\n--- Training Random Forest (Without RAG) ---")
    pipe_rf_base = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(base_num_cols, cat_cols)),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]
    )
    pipe_rf_base.fit(train_df[base_num_cols + cat_cols], y_train)
    val_prob = pipe_rf_base.predict_proba(val_df[base_num_cols + cat_cols])[:, 1]
    metrics_rf_base = evaluate_predictions(y_val, val_prob)
    metrics_rf_base.update({"model": "Random Forest", "rag_feature": False})
    ablation_records.append(metrics_rf_base)
    print(f"Validation AUC: {metrics_rf_base['auc_roc']} (CI: {metrics_rf_base['auc_ci_95']}) | F1: {metrics_rf_base['f1']} | Prec@10%: {metrics_rf_base['precision_top_10pct']} (Lift: {metrics_rf_base['lift_top_10pct']}x)")

    # -------------------------------------------------------------
    # 4. Random Forest: With RAG
    # -------------------------------------------------------------
    print("\n--- Training Random Forest (With RAG) ---")
    pipe_rf_rag = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(rag_num_cols, cat_cols)),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]
    )
    pipe_rf_rag.fit(train_df[rag_num_cols + cat_cols], y_train)
    val_prob = pipe_rf_rag.predict_proba(val_df[rag_num_cols + cat_cols])[:, 1]
    metrics_rf_rag = evaluate_predictions(y_val, val_prob)
    metrics_rf_rag.update({"model": "Random Forest", "rag_feature": True})
    ablation_records.append(metrics_rf_rag)
    joblib.dump(pipe_rf_rag, MODELS_DIR / "rf_rag.joblib")
    print(f"Validation AUC: {metrics_rf_rag['auc_roc']} (CI: {metrics_rf_rag['auc_ci_95']}) | F1: {metrics_rf_rag['f1']} | Prec@10%: {metrics_rf_rag['precision_top_10pct']} (Lift: {metrics_rf_rag['lift_top_10pct']}x)")

    # -------------------------------------------------------------
    # 5. LightGBM: Without RAG (Native Categoricals)
    # -------------------------------------------------------------
    print("\n--- Training LightGBM (Without RAG, Native Categoricals) ---")
    X_train_lgbm_base = prepare_lgbm_data(train_df, base_num_cols, cat_cols)
    X_val_lgbm_base = prepare_lgbm_data(val_df, base_num_cols, cat_cols)

    lgbm_base = LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=7.7,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_base.fit(X_train_lgbm_base, y_train)
    val_prob = lgbm_base.predict_proba(X_val_lgbm_base)[:, 1]
    metrics_lgbm_base = evaluate_predictions(y_val, val_prob)
    metrics_lgbm_base.update({"model": "LightGBM", "rag_feature": False})
    ablation_records.append(metrics_lgbm_base)
    print(f"Validation AUC: {metrics_lgbm_base['auc_roc']} (CI: {metrics_lgbm_base['auc_ci_95']}) | F1: {metrics_lgbm_base['f1']} | Prec@10%: {metrics_lgbm_base['precision_top_10pct']} (Lift: {metrics_lgbm_base['lift_top_10pct']}x)")

    # -------------------------------------------------------------
    # 6. LightGBM: With RAG (Native Categoricals)
    # -------------------------------------------------------------
    print("\n--- Training LightGBM (With RAG, Native Categoricals) ---")
    X_train_lgbm_rag = prepare_lgbm_data(train_df, rag_num_cols, cat_cols)
    X_val_lgbm_rag = prepare_lgbm_data(val_df, rag_num_cols, cat_cols)

    lgbm_rag = LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=7.7,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_rag.fit(X_train_lgbm_rag, y_train)
    val_prob = lgbm_rag.predict_proba(X_val_lgbm_rag)[:, 1]
    metrics_lgbm_rag = evaluate_predictions(y_val, val_prob)
    metrics_lgbm_rag.update({"model": "LightGBM", "rag_feature": True})
    ablation_records.append(metrics_lgbm_rag)
    joblib.dump(lgbm_rag, MODELS_DIR / "lgbm_rag.joblib")
    print(f"Validation AUC: {metrics_lgbm_rag['auc_roc']} (CI: {metrics_lgbm_rag['auc_ci_95']}) | F1: {metrics_lgbm_rag['f1']} | Prec@10%: {metrics_lgbm_rag['precision_top_10pct']} (Lift: {metrics_lgbm_rag['lift_top_10pct']}x)")

    # Save separate fitted preprocessors
    joblib.dump(pipe_lr_base.named_steps["preprocessor"], PREPROCESSOR_BASE_PATH)
    joblib.dump(pipe_rf_rag.named_steps["preprocessor"], PREPROCESSOR_RAG_PATH)
    print(f"Saved base preprocessor to {PREPROCESSOR_BASE_PATH}")
    print(f"Saved RAG preprocessor to {PREPROCESSOR_RAG_PATH}")

    # Persist intermediate ablation table
    with open(METRICS_DIR / "ablation_table.json", "w") as f:
        json.dump(ablation_records, f, indent=2)

    return ablation_records


if __name__ == "__main__":
    train_and_evaluate_all_models()
