"""Choose the note retriever (notes v1/v2 x minilm/bge_small) on TRAIN out-of-fold validity only.

Criterion: univariate AUC of rag_note_rate_k20 against the TRAIN label, using 5-fold
patient-grouped out-of-fold retrieval. The validation and test splits are never read here.
Writes reports/note_rag/DECISION.md and reports/note_rag/selected_retriever.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import note_rag as nr
from src.note_rag_config import (
    MODEL_KEYS,
    NOTE_RAG_DATA_DIR,
    NOTE_RAG_REPORT_DIR,
    PRIMARY_FEATURE,
    VARIANTS,
    embedding_name,
)
from src.note_rag_io import load_split_meta, meta_arrays

SELECTION_PATH = NOTE_RAG_REPORT_DIR / "selected_retriever.json"
DECISION_PATH = NOTE_RAG_REPORT_DIR / "DECISION.md"


def evaluate_candidate(emb_dir: Path, model_key: str, variant: str, train: Dict[str, Any]) -> Dict:
    emb = nr.load_embeddings(emb_dir, model_key, variant, "train", train["ids"])
    result = nr.build_train_features(emb, train["patients"], train["y"])
    auc = nr.univariate_auc(train["y"], result.frame[PRIMARY_FEATURE].to_numpy())
    return {"model_key": model_key, "variant": variant, "oof_auc_rate_k20": auc, "rows": len(emb)}


def render_decision(rows: List[Dict], chosen: Dict) -> str:
    lines = [
        "# Note retriever decision",
        "",
        "Criterion: univariate TRAIN out-of-fold AUC of `rag_note_rate_k20` "
        "(5-fold patient-grouped retrieval, same-patient neighbors excluded).",
        "Validation and test data were not read by the selection script.",
        "",
        "| Notes | Embedding model | TRAIN OOF AUC | Rows |",
        "|:---|:---|:---:|:---:|",
    ]
    for r in sorted(rows, key=lambda r: -r["oof_auc_rate_k20"]):
        lines.append(
            f"| {r['variant']} | {r['model_key']} | {r['oof_auc_rate_k20']:.4f} | {r['rows']} |"
        )
    lines += [
        "",
        f"Selected: notes {chosen['variant']} with {chosen['model_key']} "
        f"(highest TRAIN out-of-fold AUC, {chosen['oof_auc_rate_k20']:.4f}).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-dir", type=Path, default=NOTE_RAG_DATA_DIR)
    args = parser.parse_args()

    train = meta_arrays(load_split_meta("train"))
    rows: List[Dict] = []
    for model_key in MODEL_KEYS:
        for variant in VARIANTS:
            if not (args.emb_dir / embedding_name(model_key, variant, "train")).exists():
                print(f"skip {model_key}/{variant}: no TRAIN embeddings", flush=True)
                continue
            print(f"evaluating {model_key}/{variant} ...", flush=True)
            rows.append(evaluate_candidate(args.emb_dir, model_key, variant, train))
    if not rows:
        print("No candidate embeddings found; nothing selected.")
        return 1
    chosen = max(rows, key=lambda r: r["oof_auc_rate_k20"])
    NOTE_RAG_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps({"selected": chosen, "candidates": rows}, indent=2))
    DECISION_PATH.write_text(render_decision(rows, chosen), encoding="utf-8")
    print(f"Selected notes {chosen['variant']} / {chosen['model_key']}; see {DECISION_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
