import json
from pathlib import Path

nb_path = Path("notebooks/stages/stage6.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "# 5: Transformer (Without RAG)" in src and "from src.evaluate import evaluate_predictions" in src:
            new_source = []
            for line in cell["source"]:
                if "from src.evaluate import evaluate_predictions" in line:
                    new_source.append("from src.evaluate import evaluate_predictions\n")
                elif "db_m = evaluate_predictions(y_db, p_db, n_bootstraps=500, seed=42)" in line:
                    new_source.append("db_m = evaluate_predictions(y_db, p_db, n_bootstraps=500, seed=42)\n")
                elif '"AUC-ROC": f"{db_m[\'auc_roc\']:.4f}",' in line:
                    new_source.append('    "AUC-ROC": f"{db_m[\'auc_roc\']:.4f}",\n')
                elif '"95% CI": f"[{db_m[\'auc_ci_95\'][0]:.4f}, {db_m[\'auc_ci_95\'][1]:.4f}]",' in line:
                    new_source.append('    "95% CI": f"[{db_m[\'auc_ci_95\'][0]:.4f}, {db_m[\'auc_ci_95\'][1]:.4f}]",\n')
                elif '"F1-Score": f"{db_m[\'f1\']:.4f}",' in line:
                    new_source.append('    "F1-Score": f"{db_m[\'f1\']:.4f}",\n')
                elif '"ECE": f"{db_m[\'ece\']:.4f}",' in line:
                    new_source.append('    "ECE": f"{db_m[\'ece\']:.4f}",\n')
                elif "db_rag_m = evaluate_predictions(y_rag, p_rag, n_bootstraps=500, seed=42)" in line:
                    new_source.append("    db_rag_m = evaluate_predictions(y_rag, p_rag, n_bootstraps=500, seed=42)\n")
                elif '"AUC-ROC": f"{db_rag_m[\'auc_roc\']:.4f}",' in line:
                    new_source.append('        "AUC-ROC": f"{db_rag_m[\'auc_roc\']:.4f}",\n')
                elif '"95% CI": f"[{db_rag_m[\'auc_ci_95\'][0]:.4f}, {db_rag_m[\'auc_ci_95\'][1]:.4f}]",' in line:
                    new_source.append('        "95% CI": f"[{db_rag_m[\'auc_ci_95\'][0]:.4f}, {db_rag_m[\'auc_ci_95\'][1]:.4f}]",\n')
                elif '"F1-Score": f"{db_rag_m[\'f1\']:.4f}",' in line:
                    new_source.append('        "F1-Score": f"{db_rag_m[\'f1\']:.4f}",\n')
                elif '"ECE": f"{db_rag_m[\'ece\']:.4f}",' in line:
                    new_source.append('        "ECE": f"{db_rag_m[\'ece\']:.4f}",\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print("Successfully fixed indentation in stage6.ipynb cell 3!")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Saved stage6.ipynb.")
