import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import RANDOM_SEED


def compute_bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """Compute empirical 95% bootstrap confidence interval for AUC-ROC."""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        # Resample must contain at least one positive and one negative
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        base_score = float(roc_auc_score(y_true, y_prob))
        return base_score, base_score

    lower = float(np.percentile(bootstrapped_scores, 100 * (alpha / 2)))
    upper = float(np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k_ratio: float = 0.10,
) -> Tuple[float, float, float]:
    """Compute Precision@top-k%, Lift@top-k%, and Recall@top-k% by patient capacity."""
    n_total = len(y_true)
    k_count = max(1, int(n_total * top_k_ratio))

    order = np.argsort(y_prob)[::-1]
    top_indices = order[:k_count]

    tp = np.sum(y_true[top_indices] == 1)
    precision_top_k = float(tp / k_count)

    total_positives = np.sum(y_true == 1)
    recall_top_k = float(tp / total_positives) if total_positives > 0 else 0.0

    prevalence = float(np.mean(y_true))
    lift = float(precision_top_k / prevalence) if prevalence > 0 else 1.0

    return round(precision_top_k, 4), round(lift, 2), round(recall_top_k, 4)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    top_k_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Comprehensive evaluation computing AUC-ROC, 95% bootstrap CI, F1, Precision@top-10% and Lift."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    ci_lower, ci_upper = compute_bootstrap_auc_ci(
        y_true_arr, y_prob_arr, n_bootstraps=n_bootstraps, seed=seed
    )

    prec_k, lift_k, recall_k = compute_precision_at_k(
        y_true_arr, y_prob_arr, top_k_ratio=top_k_ratio
    )

    # Threshold optimization for maximum F1
    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    denom = precisions + recalls
    f1_scores = np.divide(
        2 * (precisions * recalls),
        denom,
        out=np.zeros_like(precisions),
        where=(denom > 0),
    )
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_tuned = (y_prob_arr >= best_th).astype(int)
    f1_tuned = float(f1_score(y_true_arr, y_pred_tuned, zero_division=0))
    prec_tuned = float(precision_score(y_true_arr, y_pred_tuned, zero_division=0))
    rec_tuned = float(recall_score(y_true_arr, y_pred_tuned, zero_division=0))

    return {
        "auc_roc": round(auc, 4),
        "auc_ci_95": [ci_lower, ci_upper],
        "optimal_threshold": round(best_th, 4),
        "f1": round(f1_tuned, 4),
        "precision": round(prec_tuned, 4),
        "recall": round(rec_tuned, 4),
        "precision_top_10pct": prec_k,
        "lift_top_10pct": lift_k,
        "recall_top_10pct": recall_k,
    }
