import json
from pathlib import Path

nb_path = Path("notebooks/stages/stage4.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

current_section = "Setup"
section_outputs = []

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        for line in src.splitlines():
            if line.startswith("#"):
                current_section = line.strip("# ").strip()
                break
    elif cell["cell_type"] == "code":
        outputs = cell.get("outputs", [])
        cell_text_chunks = []
        for out in outputs:
            ot = out.get("output_type")
            if ot == "stream":
                lines = out.get("text", [])
                clean_lines = [l for l in lines if not any(w in l for w in ["Warning", "warning", "findfont", "DEBUG", "INFO:"])]
                if clean_lines:
                    cell_text_chunks.append("".join(clean_lines).strip())
            elif ot in ["execute_result", "display_data"]:
                data = out.get("data", {})
                if "text/plain" in data:
                    t = data["text/plain"]
                    if "<Figure" not in str(t):
                        cell_text_chunks.append(str(t).strip())
        
        if cell_text_chunks:
            section_outputs.append((idx, current_section, "\n\n".join(cell_text_chunks)))

for cell_id, sec, text in section_outputs:
    print(f"\n{'='*75}")
    print(f"[{sec}] (Cell {cell_id})")
    print(f"{'='*75}")
    print(text)
