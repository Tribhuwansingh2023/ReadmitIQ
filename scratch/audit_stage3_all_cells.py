import json
from pathlib import Path

nb_path = Path("notebooks/stages/stage3.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    ct = cell["cell_type"]
    src = "".join(cell.get("source", []))
    first_line = src.splitlines()[0] if src.splitlines() else "<empty>"
    print(f"Cell {idx:02d} [{ct:8s}]: {first_line[:80]}")
