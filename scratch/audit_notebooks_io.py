import json
from pathlib import Path

notebooks = [
    "notebooks/01_eda_and_data_preparation.ipynb",
    "notebooks/colab_embed_notes.ipynb",
    "notebooks/colab_transformer_training.ipynb",
    "notebooks/colab_transformer_training_noterag.ipynb",
    "notebooks/xai.ipynb",
    "notebooks/stages/stage1.ipynb",
    "notebooks/stages/stage2.ipynb",
    "notebooks/stages/stage3.ipynb",
    "notebooks/stages/stage4.ipynb",
    "notebooks/stages/stage5.ipynb",
    "notebooks/stages/stage6.ipynb",
]

report = {}

for nb_path in notebooks:
    p = Path(nb_path)
    if not p.exists():
        report[nb_path] = {"error": "File not found"}
        continue
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        report[nb_path] = {"error": str(e)}
        continue
    
    cells = data.get("cells", [])
    nb_info = {
        "total_cells": len(cells),
        "code_cells": 0,
        "markdown_cells": 0,
        "imports": [],
        "reads": [],
        "writes": [],
        "auc_calls": [],
        "cells_summary": []
    }
    
    for i, c in enumerate(cells):
        cell_type = c.get("cell_type", "")
        source = "".join(c.get("source", []))
        if cell_type == "markdown":
            nb_info["markdown_cells"] += 1
        elif cell_type == "code":
            nb_info["code_cells"] += 1
            # Check for reads
            for kw in [".read_parquet", ".read_csv", "read_index", "joblib.load", "torch.load", "json.load", "open("]:
                if kw in source:
                    nb_info["reads"].append((i, kw))
            # Check for writes
            for kw in [".to_parquet", ".to_csv", "write_index", "joblib.dump", "torch.save", "json.dump"]:
                if kw in source:
                    nb_info["writes"].append((i, kw))
            # Check for auc calls
            for kw in ["roc_auc_score", "compute_roc_auc", "auc(", "auc_roc"]:
                if kw in source:
                    nb_info["auc_calls"].append((i, kw))
    report[nb_path] = nb_info

print(json.dumps(report, indent=2))
