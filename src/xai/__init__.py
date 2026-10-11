"""ReadmitIQ Explainable AI (XAI) package.

Provides local patient risk factor attributions, group-level fairness audits,
evidence agreement scoring, and global explanation quality metrics.
"""

from src.xai.config import (
    FEATURE_GROUPS,
    FEATURE_MODIFIABILITY,
    NOT_RECORDED_PATTERNS,
    PATHS,
    RISK_TIERS,
    get_feature_group,
    is_not_recorded,
    wilson_score_interval,
)

__all__ = [
    "FEATURE_GROUPS",
    "FEATURE_MODIFIABILITY",
    "NOT_RECORDED_PATTERNS",
    "PATHS",
    "RISK_TIERS",
    "get_feature_group",
    "is_not_recorded",
    "wilson_score_interval",
]
