import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

payload = {
    "age": "[70-80)",
    "gender": "Male",
    "race": "AfricanAmerican",
    "time_in_hospital": 8,
    "num_lab_procedures": 65,
    "num_procedures": 2,
    "num_medications": 24,
    "number_outpatient": 1,
    "number_emergency": 2,
    "number_inpatient": 3,
    "number_diagnoses": 9,
    "admission_type_group": "Emergency",
    "admission_source_group": "Emergency Room",
    "discharge_disposition_group": "Transferred",
    "medical_specialty": "Cardiology",
    "payer_code": "Medicare",
    "diag_1_group": "Circulatory",
    "diag_2_group": "Diabetes",
    "diag_3_group": "Respiratory",
    "max_glu_serum": ">200",
    "A1Cresult": ">8",
    "diabetesMed": "Yes",
    "change": "Ch",
    "insulin": "Up",
    "metformin": "Steady",
}

resp = client.post("/v1/predict", json=payload)
print("HTTP Status Code:", resp.status_code)
data = resp.json()

print("--- MODEL OUTPUT & SCORING ---")
print("Raw Continuous LightGBM Score:", round(data["raw_score"], 4))
print("Calibrated 30-Day Risk Probability:", round(data["calibrated_prob"], 4))
print("Assigned Hospital Capacity Risk Tier:", data["tier"])

print("\n--- CLINICAL SHAP EXPLANATION (RISK DRIVERS) ---")
print("Top Positive Risk Drivers (Increasing Risk):")
for f in data["shap_explanation"]["top_risk_increasing"][:3]:
    print(f"  • {f['feature']} ({f['explanation']}) -> SHAP attribution: {f['shap_value']:+.4f}")

print("\nTop Protective Factors (Reducing Risk):")
for f in data["shap_explanation"]["top_risk_reducing"][:2]:
    print(f"  • {f['feature']} ({f['explanation']}) -> SHAP attribution: {f['shap_value']:+.4f}")

print("\n--- STRUCTURED RAG PEER RETRIEVAL ---")
print("Neighborhood summary:", data.get("neighbor_summary"))

print("\n--- CARE-TRANSITION SUGGESTIONS ---")
suggestions = data.get("suggested_interventions", {}).get("suggestions", [])
print(f"Triggered Interventions Count: {len(suggestions)}")
for s in suggestions[:3]:
    print(f"  [{s['priority'].upper()}] {s['action']} (Triggered by: {s['trigger_driver']})")
