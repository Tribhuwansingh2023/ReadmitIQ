import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200, healthy status, and all artifacts loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "git_commit_hash" in data
    assert data["artifacts_loaded"]["model"] is True
    assert data["artifacts_loaded"]["retriever_faiss"] is True
    assert data["artifacts_loaded"]["calibrator"] is True


def test_predict_endpoint_valid():
    """Verify /predict returns calibrated probability, risk tier, SHAP reasons, and neighbors."""
    payload = {
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
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Other",
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "discharge_disposition_group": "Home",
        "medical_specialty": "InternalMedicine",
        "payer_code": "Medicare",
        "A1Cresult": ">8",
        "max_glu_serum": "None",
        "insulin": "Steady",
        "metformin": "Steady",
        "diabetesMed": "Yes",
        "change": "No",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["calibrated_risk_probability"] <= 1.0
    assert data["risk_tier"] in ("Low", "Medium", "High", "Critical")
    assert "decision_threshold" in data
    assert len(data["top_risk_increasing_factors"]) > 0
    assert "similar_patients_summary" in data
    assert data["similar_patients_summary"]["k_neighbors"] == 20
    assert 0.0 <= data["similar_patients_summary"]["neighbor_readmission_rate"] <= 1.0


def test_predict_endpoint_invalid_input():
    """Verify /predict rejects out-of-range inputs with HTTP 422."""
    payload = {
        "time_in_hospital": 0,  # Invalid: ge=1 required
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_similar_endpoint():
    """Verify /similar returns historical matches and neighborhood rate."""
    payload = {
        "patient_features": {
            "time_in_hospital": 3,
            "number_inpatient": 1,
            "diag_1_group": "Circulatory",
        },
        "k": 10,
    }
    response = client.post("/similar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["k_requested"] == 10
    assert len(data["similar_encounters"]) == 10
    assert 0.0 <= data["neighbor_readmission_rate"] <= 1.0
    first = data["similar_encounters"][0]
    assert "similarity_score" in first
    assert "readmitted_30d" in first


def test_explain_endpoint():
    """Verify /explain returns SHAP factor breakdowns."""
    payload = {
        "time_in_hospital": 4,
        "number_inpatient": 2,
        "number_emergency": 0,
        "diag_1_group": "Circulatory",
    }
    response = client.post("/explain?top_k=3", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_increasing" in data
    assert "top_risk_reducing" in data
    assert len(data["top_risk_increasing"]) <= 3


def test_cohort_replay_endpoint():
    """Verify /cohort returns retrospective replay KPIs on held-out test split."""
    response = client.get("/cohort?capacity=0.10")
    assert response.status_code == 200
    data = response.json()
    assert "Retrospective Replay" in data["mode"]
    assert data["capacity_fraction"] == 0.10
    assert data["total_test_encounters"] == 14903
    assert data["flagged_encounters_count"] == 1490
    assert data["true_readmissions_caught"] > 0
    assert data["precision"] > 0.15
    assert data["lift_vs_random"] > 1.5
    assert len(data["top_patients"]) > 0


def test_metrics_endpoint():
    """Verify /metrics serves saved JSON artifacts without calculation."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "ablation_table" in data
    assert "test_results" in data
    assert "fairness_audit" in data
    assert "drift_metrics" in data
    assert "frozen_selection" in data
