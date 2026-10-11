import json
from pathlib import Path

nb_path = Path("notebooks/stages/stage3.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

for idx in [0, 6, 7, 10, 11]:
    print(f"\n{'='*40} CELL {idx} ({nb['cells'][idx]['cell_type']}) {'='*40}")
    print("".join(nb["cells"][idx]["source"]))
