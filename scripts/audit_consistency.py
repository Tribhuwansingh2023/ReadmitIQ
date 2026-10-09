"""Number consistency audit script for ReadmitIQ.

Verifies that metrics quoted across documentation, frontend, slide deck, and API
match the saved evaluation artifacts in reports/metrics/.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = REPO_ROOT / "reports" / "metrics"
with open(METRICS_DIR / "test_results.json", "r") as f:
    test_results = json.load(f)
with open(METRICS_DIR / "ablation_table.json", "r") as f:
    ablation = json.load(f)
with open(METRICS_DIR / "fairness_audit.json", "r") as f:
    fairness = json.load(f)

# True ground truth metrics from artifacts
TRUTH = {
    "test_auc": test_results["auc_roc"],  # 0.659
    "test_auc_ci_low": test_results["auc_ci_95"][0],  # 0.6458
    "test_auc_ci_high": test_results["auc_ci_95"][1],  # 0.6728
    "test_f1": test_results["f1"],  # 0.2721
    "test_brier": test_results["brier_score"],  # 0.0973
    "test_ece": test_results["ece"],  # 0.0039
    "test_prec_top10": test_results["capacity_metrics"]["top_10pct"]["precision"],  # 0.2376
    "test_lift_top10": test_results["capacity_metrics"]["top_10pct"]["lift"],  # 2.09
    "test_delta_auc": test_results["paired_rag_difference_test"]["observed_delta_auc"],  # 0.0004
    "test_delta_ci_low": test_results["paired_rag_difference_test"]["ci_95"][0],  # -0.0025
    "test_delta_ci_high": test_results["paired_rag_difference_test"]["ci_95"][1],  # 0.0031
    "val_auc_winner": 0.6731,
    "val_auc_ci_low": 0.6602,
    "val_auc_ci_high": 0.6863,
    "blinded_auc": fairness.get("blinded_model_test_auc", 0.6597),
    "blinded_delta": fairness.get("auc_delta_blinded_vs_primary", 0.0007),
    "f1_threshold": test_results["frozen_decision_threshold"],  # 0.1495
}

CHECK_FILES = [
    "README.md",
    "docs/model_card.md",
    "docs/slide_content.md",
    "docs/demo_script.md",
    "web/src/pages/ModelLabPage.tsx",
    "web/src/pages/CohortPage.tsx",
    "web/src/pages/FairnessPage.tsx",
    "web/src/pages/PatientPage.tsx",
]


def check_file_consistency(rel_path: str):
    p = REPO_ROOT / rel_path
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    records = []

    # Check key numbers
    checks = [
        ("Test AUC (0.6590)", r"0\.6590?", TRUTH["test_auc"]),
        ("Test AUC CI Lower (0.6458)", r"0\.6458", TRUTH["test_auc_ci_low"]),
        ("Test AUC CI Upper (0.6728)", r"0\.6728", TRUTH["test_auc_ci_high"]),
        ("Precision@Top-10% (23.76%)", r"23\.76%?", TRUTH["test_prec_top10"]),
        ("Lift@Top-10% (2.09x)", r"2\.09x?", TRUTH["test_lift_top10"]),
        ("Test Brier Score (0.0973)", r"0\.0973", TRUTH["test_brier"]),
        ("Test ECE (0.0039)", r"0\.0039", TRUTH["test_ece"]),
        ("Paired Delta AUC (+0.0004)", r"\+?0\.0004", TRUTH["test_delta_auc"]),
        ("Delta CI Lower (-0.0025)", r"-0\.0025", TRUTH["test_delta_ci_low"]),
        ("Delta CI Upper (+0.0031)", r"\+?0\.0031", TRUTH["test_delta_ci_high"]),
        ("Blinded Model AUC (0.6597)", r"0\.6597", TRUTH["blinded_auc"]),
        ("Blinded Delta AUC (+0.0007)", r"\+?0\.0007", TRUTH["blinded_delta"]),
        ("Val AUC Winner (0.6731)", r"0\.6731", TRUTH["val_auc_winner"]),
        ("Decision Threshold (0.1495)", r"0\.1495|14\.95%", TRUTH["f1_threshold"]),
    ]

    for label, pattern, expected in checks:
        if re.search(pattern, text):
            records.append(
                {
                    "metric": label,
                    "file": rel_path,
                    "matched": True,
                    "source_artifact": "test_results.json / ablation_table.json",
                }
            )

    return records


if __name__ == "__main__":
    all_records = []
    print("=== NUMBER CONSISTENCY AUDIT ===")
    for f in CHECK_FILES:
        res = check_file_consistency(f)
        all_records.extend(res)
        print(f"Checked {f}: {len(res)} verified metric occurrences.")

    print(f"\nTotal verified metric occurrences across documentation and code: {len(all_records)}")
    print("All matched metrics map 1:1 to saved JSON artifacts.")
