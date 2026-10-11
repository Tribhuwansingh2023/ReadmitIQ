import json
from pathlib import Path

for nb_path in sorted(list(Path("notebooks/stages").glob("stage*.ipynb")) + list(Path("notebooks").glob("*.ipynb"))):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n==================== {nb_path} ====================")
    cell_idx = 0
    for cell in nb.get("cells", []):
        cell_idx += 1
        for out in cell.get("outputs", []):
            out_type = out.get("output_type")
            if out_type == "error":
                print(f"  [Cell {cell_idx}] ERROR: {out.get('ename')}: {out.get('evalue')}")
            elif out_type == "stream":
                stream_name = out.get("name", "")
                text = "".join(out.get("text", []))
                for line in text.splitlines():
                    lower = line.lower()
                    if any(kw in lower for kw in ["warning", "error", "failed", "failure", "traceback", "exception"]):
                        print(f"  [Cell {cell_idx}] {stream_name.upper()}: {line.strip()[:140]}")

# Also inspect stage6.ipynb cell 26
with open("notebooks/stages/stage6.ipynb", "r", encoding="utf-8") as f:
    s6 = json.load(f)
print("\n--- STAGE6 CELL 26 CODE & OUTPUT ---")
c26 = s6["cells"][25] # 0-indexed 25 is cell 26
print("SOURCE:\n", "".join(c26.get("source", [])))
print("OUTPUTS:\n", json.dumps(c26.get("outputs", []), indent=2))
