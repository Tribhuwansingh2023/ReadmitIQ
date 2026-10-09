import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_LIST_PATH,
    METRICS_DIR,
    TEST_RAG_PARQUET,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference (expected) and target (actual) distributions."""
    exp_arr = np.asarray(expected)
    act_arr = np.asarray(actual)

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    # Determine if continuous or categorical
    if np.issubdtype(exp_arr.dtype, np.number):
        # Continuous numeric: decile binning based on expected
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(exp_arr, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        # Deduplicate identical bin edges
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        exp_counts, _ = np.histogram(exp_arr, bins=bins)
        act_counts, _ = np.histogram(act_arr, bins=bins)
    else:
        # Categorical: count per distinct level
        all_cats = np.unique(np.concatenate([exp_arr.astype(str), act_arr.astype(str)]))
        exp_counts = np.array([np.sum(exp_arr == c) for c in all_cats])
        act_counts = np.array([np.sum(act_arr == c) for c in all_cats])

    exp_pct = (exp_counts / len(exp_arr)) + epsilon
    act_pct = (act_counts / len(act_arr)) + epsilon

    # Normalize back to sum to 1
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_val, 4))


def classify_drift_severity(psi_value: float) -> str:
    """Classify PSI severity according to industry monitoring standards."""
    if psi_value < 0.10:
        return "Minimal (Stable)"
    elif psi_value < 0.25:
        return "Moderate (Monitor)"
    else:
        return "High (Alert)"


def compute_drift_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    train_scores: Optional[np.ndarray] = None,
    val_scores: Optional[np.ndarray] = None,
    test_scores: Optional[np.ndarray] = None,
) -> Dict:
    """Compute PSI for all features and model score distributions across Train-Val and Train-Test."""
    results = {
        "features": {},
        "scores": {},
        "summary": {
            "total_features_monitored": len(feature_cols),
            "val_moderate_drift_count": 0,
            "val_high_drift_count": 0,
            "test_moderate_drift_count": 0,
            "test_high_drift_count": 0,
        },
    }

    for col in feature_cols:
        tr_val = train_df[col].values
        va_val = val_df[col].values
        te_val = test_df[col].values

        psi_val = calculate_psi(tr_val, va_val)
        psi_test = calculate_psi(tr_val, te_val)

        status_val = classify_drift_severity(psi_val)
        status_test = classify_drift_severity(psi_test)

        if "Moderate" in status_val:
            results["summary"]["val_moderate_drift_count"] += 1
        elif "High" in status_val:
            results["summary"]["val_high_drift_count"] += 1

        if "Moderate" in status_test:
            results["summary"]["test_moderate_drift_count"] += 1
        elif "High" in status_test:
            results["summary"]["test_high_drift_count"] += 1

        results["features"][col] = {
            "psi_train_vs_val": psi_val,
            "status_train_vs_val": status_val,
            "psi_train_vs_test": psi_test,
            "status_train_vs_test": status_test,
        }

    # Model Score Drift
    if train_scores is not None and val_scores is not None and test_scores is not None:
        score_psi_val = calculate_psi(train_scores, val_scores)
        score_psi_test = calculate_psi(train_scores, test_scores)
        results["scores"] = {
            "psi_train_vs_val": score_psi_val,
            "status_train_vs_val": classify_drift_severity(score_psi_val),
            "psi_train_vs_test": score_psi_test,
            "status_train_vs_test": classify_drift_severity(score_psi_test),
        }

    return results


def run_drift_analysis(output_path: Optional[Path] = None) -> Dict:
    """Run full drift monitoring analysis and save results."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = METRICS_DIR / "drift_metrics.json"

    with open(FEATURE_LIST_PATH, "r") as f:
        f_list = json.load(f)

    cols = f_list["base_features"]
    print("Loading data splits for drift analysis...")
    train_df = pd.read_parquet(TRAIN_RAG_PARQUET, columns=cols)
    val_df = pd.read_parquet(VAL_RAG_PARQUET, columns=cols)
    test_df = pd.read_parquet(TEST_RAG_PARQUET, columns=cols)

    drift_report = compute_drift_metrics(train_df, val_df, test_df, cols)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)

    print(f"Drift monitoring analysis complete. Results saved to {output_path}")
    print(
        f"Summary: Test Moderate={drift_report['summary']['test_moderate_drift_count']}, Test High={drift_report['summary']['test_high_drift_count']}"
    )
    return drift_report


if __name__ == "__main__":
    run_drift_analysis()
