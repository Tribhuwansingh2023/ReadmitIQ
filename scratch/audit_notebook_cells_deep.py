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

for nb_p in notebooks:
    p = Path(nb_p)
    if not p.exists(): continue
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {nb_p} (cells: {len(nb['cells'])}) ===")
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            for phrase in ["auc", "roc", "evaluate_predictions", "compute_roc_auc", "hardcode", "test_rag"]:
                if phrase in src.lower():
                    # print snippet
                    first_line = cell.get("source", [""])[0].strip() if cell.get("source") else ""
                    print(f"  Cell {idx}: contains '{phrase}' -> {first_line[:60]}")
                    break
