import json

with open("notebooks/stages/stage5.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    for t in ["pending", "render_pending_box"]:
        if t in src.lower():
            print(f"Cell {i} ({c['cell_type']}): found '{t}'")
