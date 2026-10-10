"""Build reports/metrics/six_model_grid.json and .md from the saved ablation artifacts.

Transformer predictions are read from the Colab outputs:
    without RAG : reports/colab/ or reports/colab_outputs/  distilbert_val_predictions_notes.csv
    note-RAG    : reports/note_rag/colab/distilbert_val_predictions_notes_noterag.csv
A missing file is shown as "pending".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR, METRICS_DIR, REPORTS_DIR
from src.note_rag_config import NOTE_RAG_COLAB_DIR
from src.six_model_grid import build_grid, render_markdown

NOTE_JSON = METRICS_DIR / "ablation_note_rag.json"
TRANSFORMER_FILE = "distilbert_val_predictions_notes.csv"
TRANSFORMER_NOTERAG_FILE = "distilbert_val_predictions_notes_noterag.csv"


def first_existing(paths: list) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def main() -> int:
    without = first_existing(
        [REPORTS_DIR / "colab" / TRANSFORMER_FILE, REPORTS_DIR / "colab_outputs" / TRANSFORMER_FILE]
    )
    with_rag = NOTE_RAG_COLAB_DIR / TRANSFORMER_NOTERAG_FILE
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grid(NOTE_JSON, without, with_rag, BASE_DIR)
    (METRICS_DIR / "six_model_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    (METRICS_DIR / "six_model_grid.md").write_text(render_markdown(grid), encoding="utf-8")
    pending = sum(r["status"] != "ok" for r in grid["rows"])
    print(f"Grid written: {len(grid['rows'])} rows, {pending} pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
