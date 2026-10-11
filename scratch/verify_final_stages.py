from pathlib import Path
import json

print("=== CHECKING NOTEBOOKS ===")
for nb_path in sorted(Path("notebooks/stages").glob("stage*.ipynb")):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    stderrs = []
    errors = []
    for ci, c in enumerate(nb["cells"], 1):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append(f"Cell {ci}: {o.get('ename')}")
            elif o.get("output_type") == "stream" and o.get("name") == "stderr":
                stderrs.append(f"Cell {ci}: {''.join(o.get('text', []))[:60]}")
    print(f"{nb_path.name}: errors={len(errors)}, stderr_outputs={len(stderrs)}")
    if errors or stderrs:
        print("  ", errors, stderrs)

print("\n=== CHECKING HTML REPORTS ===")
for p in sorted(Path("reports/stages").glob("stage*.html")):
    txt = p.read_text(encoding="utf-8")
    warn_count = txt.count("InconsistentVersionWarning") + txt.count("findfont") + txt.count("TqdmWarning")
    fail_table = txt.count("<td>FAIL</td>")
    pass_table = txt.count("<td>PASS</td>")
    fail_warn = txt.count("WARNING: One or more Stage")
    print(f"{p.name}: warnings={warn_count}, fail_table={fail_table}, pass_table={pass_table}, fail_warn={fail_warn}")
