"""Compute raw-score cutoffs from VALIDATION scores of the frozen model and summary distributions for score-drift PSI.

Saves output to models/final/serving_thresholds.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TRAIN_RAG_PARQUET, VAL_RAG_PARQUET
from src.train import prepare_lgbm_data


def generate_serving_thresholds():
    final_dir = MODELS_DIR / "final"
    model = joblib.load(final_dir / "model.joblib")
    with open(final_dir / "feature_list.json", "r", encoding="utf-8") as f:
        feat_meta = json.load(f)

    num_cols = feat_meta["numeric_features"]
    cat_cols = feat_meta["categorical_features"]

    print("Loading training data for distribution summary...")
    train_df = pd.read_parquet(TRAIN_RAG_PARQUET)
    X_train = prepare_lgbm_data(train_df, num_cols, cat_cols)
    train_raw = model.predict_proba(X_train)[:, 1]

    print("Loading validation data for cutoffs...")
    val_df = pd.read_parquet(VAL_RAG_PARQUET)
    X_val = prepare_lgbm_data(val_df, num_cols, cat_cols)
    val_raw = model.predict_proba(X_val)[:, 1]

    val_n = len(val_raw)
    val_sorted = np.sort(val_raw)[::-1]

    k5 = int(round(val_n * 0.05))
    k10 = int(round(val_n * 0.10))
    k20 = int(round(val_n * 0.20))

    th_5 = float(val_sorted[k5 - 1])
    th_10 = float(val_sorted[k10 - 1])
    th_20 = float(val_sorted[k20 - 1])

    # Training 10 deciles for PSI reference
    deciles = np.linspace(0, 1, 11)
    decile_bins = [float(x) for x in np.quantile(train_raw, deciles)]
    decile_bins[0] = 0.0
    decile_bins[-1] = 1.0

    # Counts in each decile for training (each should be ~10%)
    train_bin_counts, _ = np.histogram(train_raw, bins=decile_bins)
    train_bin_shares = [float(c / len(train_raw)) for c in train_bin_counts]

    val_bin_counts, _ = np.histogram(val_raw, bins=decile_bins)
    val_bin_shares = [float(c / len(val_raw)) for c in val_bin_counts]

    def dist_summary(arr: np.ndarray, name: str) -> dict:
        return {
            "name": name,
            "sample_size": int(len(arr)),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "percentiles": {
                "p1": round(float(np.percentile(arr, 1)), 6),
                "p5": round(float(np.percentile(arr, 5)), 6),
                "p10": round(float(np.percentile(arr, 10)), 6),
                "p25": round(float(np.percentile(arr, 25)), 6),
                "p50": round(float(np.percentile(arr, 50)), 6),
                "p75": round(float(np.percentile(arr, 75)), 6),
                "p90": round(float(np.percentile(arr, 90)), 6),
                "p95": round(float(np.percentile(arr, 95)), 6),
                "p99": round(float(np.percentile(arr, 99)), 6),
            },
        }

    output = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": "LightGBM + Structured RAG",
            "model_path": "models/final/model.joblib",
            "validation_source": "data/processed/val_rag.parquet",
            "training_source": "data/processed/train_rag.parquet",
        },
        "raw_score_cutoffs": {
            "top_5pct": round(th_5, 6),
            "top_10pct": round(th_10, 6),
            "top_20pct": round(th_20, 6),
        },
        "tier_definitions": {
            "Critical": f"raw_score >= {round(th_5, 6):.6f} (top 5% validation raw score)",
            "High": f"raw_score >= {round(th_10, 6):.6f} and raw_score < {round(th_5, 6):.6f} (top 5-10%)",
            "Medium": f"raw_score >= {round(th_20, 6):.6f} and raw_score < {round(th_10, 6):.6f} (top 10-20%)",
            "Low": f"raw_score < {round(th_20, 6):.6f} (bottom 80%)",
        },
        "psi_reference": {
            "decile_bin_edges": [round(b, 6) for b in decile_bins],
            "training_shares": [round(s, 6) for s in train_bin_shares],
            "validation_shares": [round(s, 6) for s in val_bin_shares],
        },
        "training_distribution": dist_summary(train_raw, "train_raw_scores"),
        "validation_distribution": dist_summary(val_raw, "val_raw_scores"),
    }

    out_file = final_dir / "serving_thresholds.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[SUCCESS] Saved serving thresholds to {out_file}")
    return output


if __name__ == "__main__":
    generate_serving_thresholds()
