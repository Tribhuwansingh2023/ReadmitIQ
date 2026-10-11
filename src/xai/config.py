"""Configuration loader and schema helpers for the ReadmitIQ Explainable AI (XAI) layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "xai.yaml"

DEFAULT_PATHS = {
    "model": "models/final/model.joblib",
    "calibrator": "models/final/calibrator.joblib",
    "feature_list": "models/final/feature_list.json",
    "serving_thresholds": "models/final/serving_thresholds.json",
    "train_data": "data/processed/train_rag.parquet",
    "val_data": "data/processed/val_rag.parquet",
    "output_dir": "reports/xai",
    "figures_dir": "reports/xai/figures",
}

XAI_CONFIG: Dict[str, Any] = {}
try:
    import yaml
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            XAI_CONFIG = yaml.safe_load(f) or {}
except Exception:
    XAI_CONFIG = {}

paths_dict = XAI_CONFIG.get("paths", DEFAULT_PATHS)
PATHS = {
    k: REPO_ROOT / paths_dict.get(k, DEFAULT_PATHS[k])
    for k in DEFAULT_PATHS
}
PATHS["output_dir"].mkdir(parents=True, exist_ok=True)
PATHS["figures_dir"].mkdir(parents=True, exist_ok=True)

SEED = XAI_CONFIG.get("seed", 42)
FEATURE_GROUPS = XAI_CONFIG.get("feature_groups", {})
FEATURE_MODIFIABILITY = XAI_CONFIG.get("feature_modifiability", {})
NOT_RECORDED_PATTERNS = XAI_CONFIG.get("not_recorded_patterns", {})
RISK_TIERS = XAI_CONFIG.get("risk_tiers", {})

# Build reverse lookup: feature -> group_id
FEATURE_TO_GROUP: Dict[str, str] = {}
for g_id, g_info in FEATURE_GROUPS.items():
    for feat in g_info.get("features", []):
        FEATURE_TO_GROUP[feat] = g_id
        # Aliases
        if feat == "rag_struct_rate_k20":
            FEATURE_TO_GROUP["rag_readmit_rate_k20"] = g_id


def get_feature_group(feature_name: str) -> Tuple[str, str, str]:
    """Return (group_id, group_name, group_rationale) for a given feature."""
    clean_name = (
        "rag_struct_rate_k20"
        if feature_name in ("rag_struct_rate_k20", "rag_readmit_rate_k20")
        else feature_name
    )
    g_id = FEATURE_TO_GROUP.get(clean_name, "other")
    if g_id in FEATURE_GROUPS:
        return g_id, FEATURE_GROUPS[g_id]["name"], FEATURE_GROUPS[g_id]["rationale"]
    return "other", "Other Clinical Features", "General inpatient encounter observation."


def is_not_recorded(feature_name: str, value: Any) -> bool:
    """Check if the feature value denotes missing, unmeasured, or not recorded information."""
    if value is None:
        return True
    val_str = str(value).strip()
    if val_str in ("Missing", "?", "Unknown", "None", "nan", "null"):
        return True
    if feature_name in NOT_RECORDED_PATTERNS:
        patterns = [str(p).lower() for p in NOT_RECORDED_PATTERNS[feature_name]]
        if val_str.lower() in patterns:
            return True
    return False


def get_not_recorded_neutral_text(feature_name: str, human_name: str) -> str:
    """Return an honest, neutral clinical statement for unrecorded features."""
    if feature_name == "payer_code":
        return "Payer coverage was not recorded; the model evaluates this administrative absence as its own category."
    elif feature_name == "medical_specialty":
        return "Admitting medical specialty was not recorded; the model handles this as an unspecified provider category."
    elif feature_name in ("A1Cresult", "a1c_tested"):
        return "HbA1c glycemic testing was not performed during this inpatient stay; baseline metabolic monitoring is unrecorded."
    elif feature_name == "max_glu_serum":
        return "Serum glucose laboratory test was not recorded during admission."
    elif feature_name == "race":
        return "Demographic racial identity was not recorded in the clinical registry."
    return f"{human_name} was not recorded; the model treats this neutral indicator as an unobserved category."


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate exact Wilson score confidence interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    # z for two-sided 95% = 1.95996
    z = 1.959964 if abs(confidence - 0.95) < 1e-3 else 2.575829
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * float(n))) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * float(n))) / float(n))) / denom
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return float(round(lower, 4)), float(round(upper, 4))
