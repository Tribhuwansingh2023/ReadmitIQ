import json
import urllib.request

base = "http://localhost:8000"

sample_patient = {
    "age": "[70-80)",
    "gender": "Female",
    "race": "Caucasian",
    "time_in_hospital": 5,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 18,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 9,
    "diag_1": "428",
    "diag_2": "250.00",
    "diag_3": "401.9",
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "metformin": "No",
    "insulin": "Up",
    "change": "Ch",
    "diabetesMed": "Yes",
}

print("=" * 60)
print("TESTING READMITIQ LIVE BACKEND & ALL MODELS")
print("=" * 60)

# 1. Health
res = urllib.request.urlopen(f"{base}/health")
health = json.loads(res.read())
print(
    f"[OK] Health: {health['status']} | Mode: {health['serve_mode']} | Model: {health['model_name']}"
)
print(f"     Artifacts loaded: {health['artifacts_loaded']}")

# 2. Predict
req = urllib.request.Request(
    f"{base}/predict",
    data=json.dumps(sample_patient).encode(),
    headers={"Content-Type": "application/json"},
)
pred = json.loads(urllib.request.urlopen(req).read())
print(
    f"[OK] Predict: Calibrated Risk = {pred['calibrated_risk_probability']*100:.2f}% | Tier = {pred['risk_tier']}"
)
print(
    f"     Decision Threshold = {pred['decision_threshold']} | Flagged = {pred['predicted_readmission']}"
)
print(
    f"     Peer Readmission Rate = {pred['similar_patients_summary']['neighbor_readmission_rate']*100:.1f}%"
)

# 3. Explain (TreeSHAP)
req = urllib.request.Request(
    f"{base}/explain",
    data=json.dumps(sample_patient).encode(),
    headers={"Content-Type": "application/json"},
)
exp = json.loads(urllib.request.urlopen(req).read())
top_inc = exp["top_risk_increasing"][0]
print(
    f"[OK] Explain (TreeSHAP): Top Risk Driver = {top_inc['feature']} (SHAP: {top_inc['shap_value']:+.4f}) - {top_inc['explanation']}"
)

# 4. Similar (FAISS k=20)
similar_payload = {"patient_features": sample_patient, "k": 20}
req = urllib.request.Request(
    f"{base}/similar",
    data=json.dumps(similar_payload).encode(),
    headers={"Content-Type": "application/json"},
)
sim = json.loads(urllib.request.urlopen(req).read())
print(
    f"[OK] Similar (FAISS): Retrieved {len(sim['similar_encounters'])} nearest clinical peers | Peer Readmit Rate = {sim['neighbor_readmission_rate']*100:.1f}%"
)

# 5. Cohort (Triage Simulation)
res = urllib.request.urlopen(f"{base}/cohort?capacity=0.10")
coh = json.loads(res.read())
print(
    f"[OK] Cohort Triage: Total Encounters = {coh['total_test_encounters']:,} | Flagged = {coh['flagged_encounters_count']:,} | Recall = {coh['recall']*100:.1f}% | Precision = {coh['precision']*100:.1f}% | Lift = {coh['lift_vs_random']:.2f}x"
)

# 6. Metrics & Model Lab
res = urllib.request.urlopen(f"{base}/metrics")
met = json.loads(res.read())
abl = met["ablation_table"]
print(
    f"[OK] Model Lab /metrics: {len(abl['ablation_results'])} ablation models | Transformer Status = {abl['transformer_status']}"
)

# 7. Frontend check
res_front = urllib.request.urlopen("http://localhost:5173")
print(
    f"[OK] Frontend UI: HTTP {res_front.status} | Content-Type = {res_front.headers.get('Content-Type')}"
)
print("=" * 60)
print("ALL BACKEND ENDPOINTS, MODELS, AND FRONTEND ARE RUNNING PERFECTLY!")
print("=" * 60)
