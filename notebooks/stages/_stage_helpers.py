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
