"""Six-model comparison grid for use case 6 (validation split).

Rows: Logistic Regression, Random Forest, Transformer, each without RAG and with note-RAG,
plus LightGBM and the all-features set labeled "extra". An entry whose artifact is missing is
reported as "pending"; nothing is ever estimated or invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.note_rag_ablation import metric_row, paired_difference

USE_CASE_MODELS = ("Logistic Regression", "Random Forest", "Transformer")
TRANSFORMER = "Transformer"
METRIC_KEYS = (
    "auc_roc",
    "auc_ci_95",
    "f1",
    "precision",
    "recall",
    "precision_top10",
    "lift_top10",
)


def relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def pending_row(model: str, role: str, rag: str, path: Path, base: Path) -> Dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "rag": rag,
        "status": "pending",
        "reason": f"artifact not found: {relative(path, base)}",
        "metrics": None,
        "retriever": None,
        "paired_delta_auc": None,
    }


def ok_row(model, role, rag, metrics, retriever, source, paired=None) -> Dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "rag": rag,
        "status": "ok",
        "metrics": {k: metrics[k] for k in METRIC_KEYS},
        "retriever": retriever,
        "source": source,
        "paired_delta_auc": paired,
    }


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"encounter_id", "readmitted_30d", "prob"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if not np.isfinite(df["prob"]).all() or df["prob"].between(0, 1).sum() != len(df):
        raise ValueError(f"{path.name}: probabilities must be finite and within [0, 1]")
    return df


def transformer_rows(without_csv: Path, with_csv: Path, retriever: Optional[str], base: Path):
    """Transformer rows computed from prediction CSVs; pending when a CSV is absent."""
    rows: List[Dict[str, Any]] = []
    without_df = read_predictions(without_csv) if without_csv.exists() else None
    with_df = read_predictions(with_csv) if with_csv.exists() else None
    if without_df is None:
        rows.append(pending_row(TRANSFORMER, "use case", "without RAG", without_csv, base))
    else:
        y, p = without_df["readmitted_30d"].to_numpy(int), without_df["prob"].to_numpy()
        metrics = metric_row(y, p, TRANSFORMER, "none", None)
        rows.append(
            ok_row(
                TRANSFORMER, "use case", "without RAG", metrics, None, relative(without_csv, base)
            )
        )
    if with_df is None:
        rows.append(pending_row(TRANSFORMER, "use case", "with note-RAG", with_csv, base))
        return rows
    y_w, p_w = with_df["readmitted_30d"].to_numpy(int), with_df["prob"].to_numpy()
    metrics = metric_row(y_w, p_w, TRANSFORMER, "A", retriever)
    paired = None
    if without_df is not None:
        if not np.array_equal(without_df["encounter_id"], with_df["encounter_id"]):
            raise ValueError("Transformer prediction files list different encounters")
        paired = paired_difference(y_w, without_df["prob"].to_numpy(), p_w)
    rows.append(
        ok_row(
            TRANSFORMER,
            "use case",
            "with note-RAG",
            metrics,
            retriever,
            relative(with_csv, base),
            paired,
        )
    )
    return rows


def tabular_rows(note_json: Path, base: Path) -> List[Dict[str, Any]]:
    """Rows for LR, RF (use case) and LightGBM (extra) from ablation_note_rag.json."""
    plan = [
        ("Logistic Regression", "use case", "none", "without RAG"),
        ("Logistic Regression", "use case", "A", "with note-RAG"),
        ("Random Forest", "use case", "none", "without RAG"),
        ("Random Forest", "use case", "A", "with note-RAG"),
        ("LightGBM", "extra", "none", "without RAG"),
        ("LightGBM", "extra", "A", "with note-RAG"),
    ]
    if not note_json.exists():
        return [pending_row(m, role, rag, note_json, base) for m, role, _, rag in plan]
    report = json.loads(note_json.read_text(encoding="utf-8"))
    index = {(r["model"], r["feature_set"]): r for r in report["results"]}
    paired = {(d["model"], d["feature_set"]): d for d in report["paired_differences"]}
    rows = []
    for model, role, key, rag in plan:
        row = index.get((model, key))
        if row is None:
            rows.append(pending_row(model, role, rag, note_json, base))
            continue
        rows.append(
            ok_row(
                model,
                role,
                rag,
                row,
                row.get("retriever"),
                relative(note_json, base),
                paired.get((model, key)),
            )
        )
    return rows


def build_grid(
    note_json: Path,
    transformer_without: Path,
    transformer_with: Path,
    base: Path,
) -> Dict[str, Any]:
    """Assemble the grid; model order follows the use case, extras last."""
    retriever = None
    if note_json.exists():
        retriever = json.loads(note_json.read_text(encoding="utf-8")).get("retriever_label")
    rows = tabular_rows(note_json, base)[:4]
    rows += transformer_rows(transformer_without, transformer_with, retriever, base)
    rows += tabular_rows(note_json, base)[4:]
    rows.sort(
        key=lambda r: (r["role"] == "extra", _model_rank(r["model"]), r["rag"] != "without RAG")
    )
    return {"split": "validation", "rows": rows}


def _model_rank(model: str) -> int:
    order = list(USE_CASE_MODELS) + ["LightGBM"]
    return order.index(model)


def _cell(row: Dict[str, Any], key: str, fmt: str = "{:.4f}") -> str:
    if row["status"] != "ok":
        return "pending"
    return fmt.format(row["metrics"][key])


def render_markdown(grid: Dict[str, Any]) -> str:
    lines = [
        "# Six-model grid, validation split",
        "",
        "All numbers are on the validation split. 'pending' means the artifact does not exist yet.",
        "",
        "| Model | RAG | Role | AUC [95% CI] | F1 | Precision | Recall | P@10% | Lift@10% "
        "| Delta AUC vs without [95% CI] | Retriever |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]
    for r in grid["rows"]:
        if r["status"] == "ok":
            low, high = r["metrics"]["auc_ci_95"]
            auc = f"{r['metrics']['auc_roc']:.4f} [{low:.4f}, {high:.4f}]"
        else:
            auc = "pending"
        paired = r.get("paired_delta_auc")
        if paired and r["rag"] != "without RAG":
            lo, hi = paired["ci_95"]
            delta = f"{paired['observed_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        else:
            delta = "-" if r["status"] == "ok" else "pending"
        lines.append(
            f"| {r['model']} | {r['rag']} | {r['role']} | {auc} | {_cell(r, 'f1')} "
            f"| {_cell(r, 'precision')} | {_cell(r, 'recall')} | {_cell(r, 'precision_top10')} "
            f"| {_cell(r, 'lift_top10', '{:.2f}')} | {delta} | {r.get('retriever') or '-'} |"
        )
    pending = [r for r in grid["rows"] if r["status"] != "ok"]
    if pending:
        lines += ["", "Pending entries:"]
        lines += [f"- {r['model']}, {r['rag']}: {r['reason']}" for r in pending]
    lines.append("")
    return "\n".join(lines)
