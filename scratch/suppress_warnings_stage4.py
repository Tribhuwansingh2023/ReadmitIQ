import json
from pathlib import Path

nb_path = Path("notebooks/stages/stage4.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

c1_lines = nb["cells"][1]["source"]
c1_str = "".join(c1_lines)
if "warnings.filterwarnings" not in c1_str:
    new_c1 = "import warnings\nwarnings.filterwarnings('ignore')\n" + c1_str
    nb["cells"][1]["source"] = [l + "\n" for l in new_c1.splitlines()]
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Added warnings suppression to Stage 4 Cell 1.")
else:
    print("Warnings filter already present in Stage 4.")
