"""Helpers for the note-RAG ablation: metric rows, reproduction check, verdicts, markdown."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluate import (
    compute_paired_bootstrap_auc_diff,
    compute_precision_at_k,
    evaluate_predictions,
)
from src.note_rag_config import FEATURE_NAMES, PRIMARY_FEATURE, SEED

TOP_FRACTIONS = (0.05, 0.10, 0.20)
REPRODUCTION_TOLERANCE = 0.0005
REPRODUCTION_METRICS = ("auc_roc", "f1", "precision_top_10pct")
FEATURE_SETS: Dict[str, Sequence[str]] = {
    "A": (PRIMARY_FEATURE,),
    "B": tuple(FEATURE_NAMES),
}


class ReproductionError(AssertionError):
    """The 'without RAG' rows do not reproduce the existing ablation table."""


def metric_row(
    y: np.ndarray, prob: np.ndarray, model: str, feature_set: str, retriever: Optional[str]
) -> Dict[str, Any]:
    """Standard harness metrics plus precision, lift and recall at top 5/10/20 percent."""
    row = evaluate_predictions(y, prob, n_bootstraps=1000, seed=SEED)
    for frac in TOP_FRACTIONS:
        precision, lift, recall = compute_precision_at_k(y, prob, top_k_ratio=frac)
        tag = f"top{int(round(frac * 100))}"
        row[f"precision_{tag}"], row[f"lift_{tag}"], row[f"recall_{tag}"] = precision, lift, recall
    row.update(
        {
            "model": model,
            "feature_set": feature_set,
            "rag_feature": feature_set != "none",
            "retriever": retriever if feature_set != "none" else None,
        }
    )
    return row


def paired_difference(y: np.ndarray, prob_without: np.ndarray, prob_with: np.ndarray) -> Dict:
    """Paired bootstrap of AUC(with) - AUC(without): 1000 resamples, seed 42."""
    diff = compute_paired_bootstrap_auc_diff(y, prob_without, prob_with, 1000, seed=SEED)
    lower, upper = diff["ci_95"]
    diff["verdict"] = verdict_from_ci(lower, upper)
    return diff


def verdict_from_ci(lower: float, upper: float) -> str:
    """'gain' only when the whole paired CI is above zero, 'loss' when it is below."""
    if lower > 0:
        return "measurable gain"
    if upper < 0:
        return "measurable loss"
    return "no measurable predictive lift"


def check_reproduction(
    without_rows: Sequence[Dict[str, Any]],
    existing_rows: Sequence[Dict[str, Any]],
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> Dict[str, Any]:
    """Compare each 'without RAG' row with the existing ablation row; raise if any gap is too big."""
    report: Dict[str, Any] = {"tolerance": tolerance, "models": {}}
    worst = 0.0
    for row in without_rows:
        match = [r for r in existing_rows if r["model"] == row["model"] and not r["rag_feature"]]
        if len(match) != 1:
            raise ReproductionError(f"no unique existing row for {row['model']}")
        gaps = {m: abs(float(row[m]) - float(match[0][m])) for m in REPRODUCTION_METRICS}
        report["models"][row["model"]] = gaps
        worst = max(worst, max(gaps.values()))
    report["max_abs_gap"] = worst
    report["passed"] = worst <= tolerance
    if not report["passed"]:
        raise ReproductionError(
            f"'without RAG' rows differ from ablation_table.json by up to {worst:.4f} "
            f"(tolerance {tolerance}); stopping"
        )
    return report


def _ci(row: Dict[str, Any]) -> str:
    low, high = row["auc_ci_95"]
    return f"{row['auc_roc']:.4f} [{low:.4f}, {high:.4f}]"


def render_markdown(report: Dict[str, Any]) -> str:
    rows, n_val = report["results"], report["rows"]["val"]
    retriever = report["retriever_label"]
    lines = [
        "# Note-RAG ablation (validation split)",
        "",
        f"Evaluated on the validation split only (n={n_val}). Test split not used. "
        f"Retriever: {retriever}. Frozen production model unchanged.",
        "",
        "Feature set A = `rag_note_rate_k20` only. Feature set B = all note-RAG features.",
        "",
        "| Model | Features | AUC [95% CI] | F1 | Precision | Recall | P@5% | P@10% | P@20% | Lift@10% |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['feature_set']} | {_ci(r)} | {r['f1']:.4f} | {r['precision']:.4f} "
            f"| {r['recall']:.4f} | {r['precision_top5']:.4f} | {r['precision_top10']:.4f} "
            f"| {r['precision_top20']:.4f} | {r['lift_top10']:.2f} |"
        )
    lines += [
        "",
        "## Paired bootstrap AUC difference (with minus without), validation split",
        "",
        "| Model | Features | Delta AUC | 95% CI | Verdict |",
        "|:---|:---:|:---:|:---:|:---|",
    ]
    for d in report["paired_differences"]:
        low, high = d["ci_95"]
        lines.append(
            f"| {d['model']} | {d['feature_set']} | {d['observed_diff']:+.4f} "
            f"| [{low:+.4f}, {high:+.4f}] | {d['verdict']} |"
        )
    repro = report["reproduction_check"]
    if repro.get("skipped"):
        lines += ["", f"Reproduction check skipped: {repro['skipped']}.", ""]
    else:
        lines += [
            "",
            f"Reproduction of existing 'without RAG' rows: max absolute gap "
            f"{repro['max_abs_gap']:.4f} (tolerance {repro['tolerance']}), passed: {repro['passed']}.",
            "",
        ]
    return "\n".join(lines)


def summarize_verdicts(paired: List[Dict[str, Any]]) -> str:
    """Plain-words overall statement computed from the paired CIs."""

    def names(kind: str) -> str:
        found = sorted(
            {f"{d['model']} (set {d['feature_set']})" for d in paired if d["verdict"] == kind}
        )
        return ", ".join(found)

    if any(d["verdict"] == "measurable gain" for d in paired):
        return "Measurable gain for: " + names("measurable gain")
    statement = "No measurable predictive lift; retrieval provides explainable precedents."
    if any(d["verdict"] == "measurable loss" for d in paired):
        statement += " A measurable loss in AUC was observed for: " + names("measurable loss")
    return statement
