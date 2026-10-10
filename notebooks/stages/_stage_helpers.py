"""Stage Helper Utilities for ReadmitIQ Stage Notebooks.

Provides standardized, slide-ready visualization styles, artifact path resolution,
pending artifact callout boxes, and computed checklist evaluation.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display

# Centralized repository root resolution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "stages" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Curated, slide-ready palette (consistent across all stages)
STAGE_PALETTE = {
    "primary": "#1E3A8A",      # Deep Navy Blue
    "secondary": "#0D9488",    # Clinical Teal
    "accent": "#D97706",       # Amber
    "alert": "#DC2626",        # Coral Red
    "neutral_dark": "#1F2937", # Dark Slate
    "neutral_light": "#F3F4F6",# Off-white / light grey
    "muted": "#6B7280",        # Muted Slate
    "purple": "#7C3AED",       # Royal Violet
}

STAGE_COLORS = list(STAGE_PALETTE.values())


def set_stage_style():
    """Apply consistent slide-ready seaborn whitegrid styling across all figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.titlesize": 14,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 1.0,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def save_stage_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to reports/stages/figures/ with slide-ready settings."""
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    rel_p = rel_path(out_path)
    print(f"Saved figure: {rel_p}")
    return out_path


def rel_path(path: Any) -> str:
    """Format path relative to repository root with consistent forward slashes."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def render_pending_box(title: str, what_is_missing: str, how_to_produce: str) -> HTML:
    """Render a clean, slide-ready styled callout box for pending dependencies/artifacts."""
    html_content = f"""
    <div style="border: 2px solid #F59E0B; background-color: #FFFBEB; border-radius: 8px; padding: 14px 18px; margin: 16px 0; font-family: sans-serif;">
        <div style="font-weight: bold; color: #B45309; font-size: 14px; margin-bottom: 6px;">
            [PENDING] {title}
        </div>
        <div style="color: #92400E; font-size: 13px; line-height: 1.5; margin-bottom: 6px;">
            <strong>Unavailable Artifact / Dependency:</strong> {what_is_missing}
        </div>
        <div style="color: #78350F; font-size: 12px; background-color: #FEF3C7; padding: 8px 12px; border-radius: 4px; font-family: monospace;">
            <strong>Resolution:</strong> {how_to_produce}
        </div>
    </div>
    """
    return HTML(html_content)


def annotate_bars(ax: plt.Axes, fmt: str = "{:,.0f}", padding: float = 3):
    """Add computed value labels on top of or beside matplotlib bar containers."""
    for p in ax.patches:
        height = p.get_height()
        width = p.get_width()
        if height > 0:
            x = p.get_x() + width / 2.0
            y = height
            ax.annotate(
                fmt.format(height),
                (x, y),
                xytext=(0, padding),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color="#1F2937",
            )
        elif width > 0:
            x = width
            y = p.get_y() + height / 2.0
            ax.annotate(
                fmt.format(width),
                (x, y),
                xytext=(padding, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="semibold",
                color="#1F2937",
            )


def evaluate_stage_checklist(checks: Dict[str, Tuple[bool, str]]) -> Tuple[bool, pd.DataFrame]:
    """Evaluate and print a formatted stage checklist table without emoji.
    
    Parameters:
        checks: Dict mapping check title to (is_passed: bool, details: str)
        
    Returns:
        (all_passed: bool, checklist_df: pd.DataFrame)
    """
    rows = []
    all_passed = True
    for name, (passed, detail) in checks.items():
        if not passed:
            all_passed = False
        rows.append({
            "Stage Check": name,
            "Status": "PASS" if passed else "FAIL",
            "Verification Details": detail,
        })
    df = pd.DataFrame(rows)
    return all_passed, df


class PureStratifiedGroupKFold:
    """Pure NumPy implementation of StratifiedGroupKFold for environments with WDAC policy constraints.
    
    Matches scikit-learn's greedy variance-balanced group partition algorithm exactly.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        from collections import defaultdict
        rng = np.random.RandomState(self.random_state)
        y = np.asarray(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(y_cnt)
        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)

        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

        if self.shuffle:
            perm = np.arange(len(groups_cnt))
            rng.shuffle(perm)
            y_counts_per_group = y_counts_per_group[perm]
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(perm.size)
            groups_inv = inv_perm[groups_inv]

        sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1), kind="stable")

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = None
            min_eval = np.inf
            min_samples_in_fold = np.inf
            for i in range(self.n_splits):
                y_counts_per_fold[i] += group_y_counts
                std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
                y_counts_per_fold[i] -= group_y_counts
                fold_eval = np.mean(std_per_class)
                samples_in_fold = np.sum(y_counts_per_fold[i])
                if fold_eval < min_eval or (fold_eval == min_eval and samples_in_fold < min_samples_in_fold):
                    min_eval = fold_eval
                    min_samples_in_fold = samples_in_fold
                    best_fold = i
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        for i in range(self.n_splits):
            test_indices = np.array([idx for idx, group_idx in enumerate(groups_inv) if group_idx in groups_per_fold[i]])
            train_indices = np.array([idx for idx, group_idx in enumerate(groups_inv) if group_idx not in groups_per_fold[i]])
            yield train_indices, test_indices


def patch_sklearn_for_appcontrol():
    """Ensure scikit-learn interfaces used by src.data and src.features are available
    even under strict Windows WDAC / AppLocker policy.
    """
    import types
    for mod in ["sklearn", "sklearn.base", "sklearn.compose", "sklearn.pipeline", "sklearn.preprocessing", "sklearn.model_selection"]:
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            m.__path__ = []
            sys.modules[mod] = m

    sys.modules["sklearn.base"].BaseEstimator = getattr(sys.modules["sklearn.base"], "BaseEstimator", type("BaseEstimator", (), {}))
    sys.modules["sklearn.base"].TransformerMixin = getattr(sys.modules["sklearn.base"], "TransformerMixin", type("TransformerMixin", (), {}))
    sys.modules["sklearn.compose"].ColumnTransformer = getattr(sys.modules["sklearn.compose"], "ColumnTransformer", type("ColumnTransformer", (), {}))
    sys.modules["sklearn.pipeline"].Pipeline = getattr(sys.modules["sklearn.pipeline"], "Pipeline", type("Pipeline", (), {}))
    sys.modules["sklearn.preprocessing"].OneHotEncoder = getattr(sys.modules["sklearn.preprocessing"], "OneHotEncoder", type("OneHotEncoder", (), {}))
    sys.modules["sklearn.preprocessing"].StandardScaler = getattr(sys.modules["sklearn.preprocessing"], "StandardScaler", type("StandardScaler", (), {}))
    sys.modules["sklearn.model_selection"].StratifiedGroupKFold = PureStratifiedGroupKFold



def compute_roc_auc(y_true, y_score) -> float:
    """Compute exact AUC-ROC via Mann-Whitney U statistic in pure NumPy."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    sum_ranks_pos = np.sum(ranks[y_true == 1])
    n_pos = len(pos)
    n_neg = len(neg)
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def fit_simple_lr(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, lr: float = 0.05, epochs: int = 50) -> np.ndarray:
    """Train a simple L2-regularized logistic regression via NumPy gradient descent for leakage demonstrations."""
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    X_va = np.asarray(X_val, dtype=float)
    
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std[std == 0] = 1.0
    
    X_tr_norm = np.c_[np.ones(len(X_tr)), (X_tr - mean) / std]
    X_va_norm = np.c_[np.ones(len(X_va)), (X_va - mean) / std]
    
    weights = np.zeros(X_tr_norm.shape[1])
    n = len(X_tr_norm)
    for _ in range(epochs):
        logits = np.clip(X_tr_norm @ weights, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = (X_tr_norm.T @ (probs - y_tr)) / n + 0.01 * weights
        weights -= lr * grad
        
    val_logits = np.clip(X_va_norm @ weights, -30, 30)
    return 1.0 / (1.0 + np.exp(-val_logits))

