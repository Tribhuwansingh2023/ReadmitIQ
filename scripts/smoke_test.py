#!/usr/bin/env python3
"""ReadmitIQ End-to-End API Smoke Test Script.

Validates that all core serving endpoints (/health, /predict, /similar, /explain,
/cohort, /metrics) respond with HTTP 200, valid JSON schemas, and clinically
sane values (probabilities in [0, 1], non-empty SHAP factors, valid risk tiers).
"""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def request_json(
    url: str, method: str = "GET", data: Dict[str, Any] = None
) -> Tuple[int, Dict[str, Any], float]:
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content), elapsed_ms
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        content = e.read().decode("utf-8")
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw": content}
        return e.code, parsed, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return 0, {"error": str(e)}, elapsed_ms


SAMPLE_PATIENT = {
    "age": "[60-70)",
    "gender": "Female",
    "race": "Caucasian",
    "time_in_hospital": 4,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 8,
    "admission_type_group": "Emergency",
    "admission_source_group": "Emergency Room",
    "discharge_disposition_group": "Home",
    "medical_specialty": "InternalMedicine",
    "payer_code": "Medicare",
    "diag_1_group": "Circulatory",
    "diag_2_group": "Diabetes",
    "diag_3_group": "Other",
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "diabetesMed": "Yes",
    "change": "Ch",
    "insulin": "Down",
    "metformin": "Steady",
}


def run_smoke_test(base_url: str) -> bool:
    print("=" * 65)
    print(f"[*] ReadmitIQ Automated API Smoke Test -- Target: {base_url}")
    print("=" * 65 + "\n")

    base_url = base_url.rstrip("/")
    passed_tests = 0
    total_tests = 6

    # 1. GET /health
    print("[1/6] Testing GET /health...")
    status, body, ms = request_json(f"{base_url}/health")
    if status == 200 and body.get("status") == "healthy" and "model_name" in body:
        print(
            f"  [PASS] ({ms:.1f}ms): Status={body.get('status')}, Model='{body.get('model_name')}'"
        )
        passed_tests += 1
    else:
        print(f"  [FAIL] ({status}): {body}")

    # 2. POST /predict
    print("\n[2/6] Testing POST /predict...")
    status, body, ms = request_json(f"{base_url}/predict", method="POST", data=SAMPLE_PATIENT)
    prob = body.get("calibrated_risk_probability")
    tier = body.get("risk_tier")
    interventions = body.get("suggested_interventions")
    if (
        status == 200
        and isinstance(prob, (int, float))
        and 0.0 <= prob <= 1.0
        and tier in ["Low", "Medium", "High", "Critical"]
    ):
        sugg_count = interventions.get("suggestions_count", 0) if interventions else 0
        print(
            f"  [PASS] ({ms:.1f}ms): Prob={prob*100:.2f}%, Tier={tier}, Interventions={sugg_count}"
        )
        passed_tests += 1
    else:
        print(f"  [FAIL] ({status}): {body}")

    # 3. POST /similar
    print("\n[3/6] Testing POST /similar...")
    status, body, ms = request_json(
        f"{base_url}/similar", method="POST", data={"patient_features": SAMPLE_PATIENT, "k": 20}
    )
    rate = body.get("neighbor_readmission_rate")
    sim_list = body.get("similar_encounters", [])
    if (
        status == 200
        and isinstance(rate, (int, float))
        and 0.0 <= rate <= 1.0
        and len(sim_list) > 0
    ):
        print(
            f"  [PASS] ({ms:.1f}ms): Neighbor Rate={rate*100:.2f}%, Retrieved={len(sim_list)} cases"
        )
        passed_tests += 1
    else:
        print(f"  [FAIL] ({status}): {body}")

    # 4. POST /explain
    print("\n[4/6] Testing POST /explain...")
    status, body, ms = request_json(
        f"{base_url}/explain?top_k=3", method="POST", data=SAMPLE_PATIENT
    )
    increasing = body.get("top_risk_increasing", [])
    reducing = body.get("top_risk_reducing", [])
    if status == 200 and len(increasing) > 0 and len(reducing) > 0:
        print(
            f"  [PASS] ({ms:.1f}ms): Increasing Factors={len(increasing)}, Reducing Factors={len(reducing)}"
        )
        passed_tests += 1
    else:
        print(f"  [FAIL] ({status}): {body}")

    # 5. GET /cohort
    print("\n[5/6] Testing GET /cohort?capacity=0.10...")
    status, body, ms = request_json(f"{base_url}/cohort?capacity=0.10")
    total_enc = body.get("total_test_encounters")
    flagged = body.get("flagged_encounters_count")
    lift = body.get("lift_vs_random")
    if status == 200 and total_enc and flagged and lift is not None:
        print(f"  [PASS] ({ms:.1f}ms): Total={total_enc}, Flagged={flagged}, Lift={lift}x")
        passed_tests += 1
    else:
        print(f"  [FAIL] ({status}): {body}")

    # 6. GET /metrics
    print("\n[6/6] Testing GET /metrics...")
    status, body, ms = request_json(f"{base_url}/metrics")
    has_ablation = "ablation_table" in body
    has_test = "test_results" in body
    has_fairness = "fairness_audit" in body
    if status == 200 and has_ablation and has_test and has_fairness:
        test_auc = body.get("test_results", {}).get("auc_roc")
        print(f"  [PASS] ({ms:.1f}ms): Ablation=OK, Test AUC={test_auc}, Fairness=OK")
        passed_tests += 1
    else:
        print(f"  [FAIL] ({status}): {body}")

    print("\n" + "=" * 65)
    print(f"Smoke Test Result: {passed_tests}/{total_tests} Tests Passed")
    print("=" * 65)
    return passed_tests == total_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReadmitIQ API Smoke Test")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="Base URL of the running API"
    )
    args = parser.parse_args()

    success = run_smoke_test(args.base_url)
    sys.exit(0 if success else 1)
