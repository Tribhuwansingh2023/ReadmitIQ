"""Train LR, RF and LightGBM with and without note-RAG features; evaluate on VALIDATION only.

Uses the existing harness (src.train, src.evaluate) with identical hyperparameters and seeds.
The 'without RAG' rows must reproduce reports/metrics/ablation_table.json within 0.0005.
Writes reports/metrics/ablation_note_rag.json and .md (new files; existing tables untouched).
The test split is never opened here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import FEATURE_LIST_PATH, METRICS_DIR, RANDOM_SEED
from src.features import build_preprocessor
from src.note_rag_ablation import (
    FEATURE_SETS,
    check_reproduction,
    metric_row,
    paired_difference,
    render_markdown,
    summarize_verdicts,
)
from src.note_rag_config import (
    ID_COL,
    LABEL_COL,
    NOTE_RAG_REPORT_DIR,
    PATIENT_COL,
    features_path,
    split_parquet,
)
from src.train import prepare_lgbm_data, tune_lightgbm, tune_random_forest

EXISTING_ABLATION = METRICS_DIR / "ablation_table.json"
OUT_JSON = METRICS_DIR / "ablation_note_rag.json"
OUT_MD = METRICS_DIR / "ablation_note_rag.md"
TUNED_PATH = NOTE_RAG_REPORT_DIR / "tuned_hparams.json"
SELECTION_PATH = NOTE_RAG_REPORT_DIR / "selected_retriever.json"
MODEL_ORDER = ("Logistic Regression", "Random Forest", "LightGBM")


def load_split(split: str, base_cols: List[str]) -> pd.DataFrame:
    """Base features + label + ids joined with the note-RAG features (aligned by row order)."""
    cols = list(dict.fromkeys(base_cols + [ID_COL, PATIENT_COL, LABEL_COL]))
    base = pd.read_parquet(split_parquet(split), columns=cols)
    feats = pd.read_parquet(features_path(split))
    if not np.array_equal(base[ID_COL].to_numpy(), feats[ID_COL].to_numpy()):
        raise ValueError(f"{split}: feature file encounter order differs from the split parquet")
    return pd.concat([base.reset_index(drop=True), feats.drop(columns=[ID_COL])], axis=1)


def tuned_hyperparameters(train_df, base_all, num_cols, cat_cols, y, groups) -> Dict[str, Any]:
    """Tune exactly like src.train does (base features only); cache the result."""
    if TUNED_PATH.exists():
        return json.loads(TUNED_PATH.read_text(encoding="utf-8"))
    rf_depth = tune_random_forest(train_df, base_all, num_cols, cat_cols, y, groups)
    lgbm_depth, lgbm_lr = tune_lightgbm(prepare_lgbm_data(train_df, num_cols, cat_cols), y, groups)
    params = {"rf_max_depth": rf_depth, "lgbm_max_depth": lgbm_depth, "lgbm_learning_rate": lgbm_lr}
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def fit_predict(
    model: str, hp: Dict[str, Any], train_df, val_df, num_cols, cat_cols, y
) -> np.ndarray:
    """Fit one model with the harness settings and return validation probabilities."""
    if model == "LightGBM":
        clf = LGBMClassifier(
            n_estimators=150,
            max_depth=hp["lgbm_max_depth"],
            learning_rate=hp["lgbm_learning_rate"],
            scale_pos_weight=7.7,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(prepare_lgbm_data(train_df, num_cols, cat_cols), y)
        return clf.predict_proba(prepare_lgbm_data(val_df, num_cols, cat_cols))[:, 1]
    if model == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=hp["rf_max_depth"],
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    pipe = Pipeline([("preprocessor", build_preprocessor(num_cols, cat_cols)), ("clf", clf)])
    pipe.fit(train_df[num_cols + cat_cols], y)
    return pipe.predict_proba(val_df[num_cols + cat_cols])[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic-data pipeline check: skip the reproduction assertion and say so in the output",
    )
    args = parser.parse_args()
    base_all = json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))["base_features"]
    retriever = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selected"]
    label = f"note embeddings {retriever['variant']} / {retriever['model_key']}"
    train_df, val_df = load_split("train", base_all), load_split("val", base_all)
    cat_cols = [c for c in base_all if not pd.api.types.is_numeric_dtype(train_df[c])]
    base_num = [c for c in base_all if pd.api.types.is_numeric_dtype(train_df[c])]
    y_train, y_val = train_df[LABEL_COL].to_numpy(int), val_df[LABEL_COL].to_numpy(int)
    print(f"train={len(train_df)} val={len(val_df)} (validation only; test not opened)", flush=True)

    hp = tuned_hyperparameters(
        train_df, base_all, base_num, cat_cols, y_train, train_df[PATIENT_COL].to_numpy()
    )
    results: List[Dict[str, Any]] = []
    paired: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        print(f"--- {model}: without RAG", flush=True)
        p_without = fit_predict(model, hp, train_df, val_df, base_num, cat_cols, y_train)
        results.append(metric_row(y_val, p_without, model, "none", None))
        for name, columns in FEATURE_SETS.items():
            print(f"--- {model}: with note-RAG set {name}", flush=True)
            num_cols = sorted(base_num + list(columns))
            p_with = fit_predict(model, hp, train_df, val_df, num_cols, cat_cols, y_train)
            results.append(metric_row(y_val, p_with, model, name, label))
            diff = paired_difference(y_val, p_without, p_with)
            diff.update({"model": model, "feature_set": name})
            paired.append(diff)

    existing = (
        json.loads(EXISTING_ABLATION.read_text(encoding="utf-8"))["ablation_results"]
        if EXISTING_ABLATION.exists()
        else []
    )
    if args.dry_run:
        repro = {"skipped": "dry run on synthetic data", "max_abs_gap": None, "passed": None}
    else:
        repro = check_reproduction([r for r in results if r["feature_set"] == "none"], existing)
    report = {
        "split": "validation",
        "rows": {"train": len(train_df), "val": len(val_df)},
        "test_split_touched": False,
        "retriever": retriever,
        "retriever_label": label,
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "hyperparameters": hp,
        "results": results,
        "paired_differences": paired,
        "overall_statement": summarize_verdicts(paired),
        "reproduction_check": repro,
        "dry_run": args.dry_run,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(report["overall_statement"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
