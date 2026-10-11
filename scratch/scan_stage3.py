import json
from pathlib import Path

nb_path = Path("notebooks/stages/stage3.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    text = "".join(cell.get("source", []))
    for term in ["pending", "PENDING", "render_pending_box", "Subsample", "colab", "Colab", "coll.query", "chroma"]:
        if term in text:
            print(f"Cell {i} ({cell['cell_type']}) has '{term}':")
            for line in text.splitlines():
                if term in line:
                    print(f"    {line.strip()[:100]}")
            break
