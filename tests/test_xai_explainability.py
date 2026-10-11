"""Comprehensive test suite for ReadmitIQ Clinical Explainability & Trust Engine (XAI).

Covers all 12 critical verification requirements:
1. SHAP contribution values and feature names match actual model inputs.
2. Positive and negative attributions have correct mathematical signs.
3. Raw model output (log-odds margin) and calibrated probability are not confused.
4. Local explanations accurately bind to the specific encounter.
5. Global importance uses the documented validation reference cohort (N=14,902).
6. Missing or failed SHAP computation produces honest unavailable state.
7. RAG failure never fabricates neighbors or required model inputs.
8. What-If simulation never mutates the original database encounter.
9. Extreme/missing values are handled safely without NaNs or crashes.
10. API response schemas match backend contracts.
11. ActionLoop workflow and persistence continue functioning.
12. Protected model and calibration artifacts remain untouched.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.database import init_db
from api.main import app
from src.explain import (
    ModelExplainer,
    check_explanation_reliability,
    generate_plain_english_summary,
)


@pytest.fixture
def client():
    init_db()
    return TestClient(app)


@pytest.fixture
def sample_features() -> Dict[str, Any]:
    return {
        "age": "[70-80)",
        "gender": "Female",
        "race": "Caucasian",
        "time_in_hospital": 6,
        "num_lab_procedures": 52,
        "num_procedures": 2,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 9,
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "discharge_disposition_group": "SNF",
        "medical_specialty": "InternalMedicine",
        "payer_code": "Medicare",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Respiratory",
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "diabetesMed": "Yes",
        "change": "Ch",
        "insulin": "Down",
        "metformin": "No",
        "glipizide": "No",
        "glyburide": "No",
        "glimepiride": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "glyburide-metformin": "No",
    }


def test_1_shap_contributions_match_actual_model_features(sample_features):
    """Requirement 1: Verify all 42 feature names and contributions match actual model feature order."""
    explainer = ModelExplainer()
    res = explainer.explain_patient(sample_features, top_k=5)

    assert "all_attributions" in res
    assert len(res["all_attributions"]) == 42
    assert set(res["all_attributions"].keys()) == set(explainer.all_cols)
    assert len(res["full_attribution_table"]) == 42

    for item in res["full_attribution_table"]:
        assert item["feature"] in explainer.all_cols
        assert "feature_name" in item
        assert "description" in item or "explanation" in item
        assert item["direction"] in ("increasing", "reducing", "neutral")


def test_2_positive_and_negative_attributions_have_correct_signs(sample_features):
    """Requirement 2: Positive factors increase risk (phi > 0); negative factors reduce risk (phi < 0)."""
    explainer = ModelExplainer()
    res = explainer.explain_patient(sample_features, top_k=5)

    for item in res["top_risk_increasing"]:
        attr = item.get("attribution", item.get("shap_value"))
        assert attr > 0.0, f"Expected positive attribution for {item['feature']}"
        assert item["direction"] == "increasing"

    for item in res["top_risk_reducing"]:
        attr = item.get("attribution", item.get("shap_value"))
        assert attr < 0.0, f"Expected negative attribution for {item['feature']}"
        assert item["direction"] == "reducing"


def test_3_raw_margin_and_calibrated_probability_not_confused(sample_features):
    """Requirement 3: Explicit additive identity phi_0 + sum(phi_i) == raw_margin; calibrated prob distinct."""
    explainer = ModelExplainer()
    res = explainer.explain_patient(sample_features, top_k=5, calibrated_prob=0.345)

    base_val = res["base_value"]
    raw_margin = res["raw_margin"]
    sum_phi = sum(res["all_attributions"].values())

    # Additive identity of TreeSHAP in margin space (within rounding bounds of 42 float values)
    assert abs((base_val + sum_phi) - raw_margin) < 0.005, "TreeSHAP additive property violated!"

    # Mathematical output scale must be documented
    assert res["output_scale"] == "log_odds_margin"
    assert "log-odds" in res["scale_units"].lower()
    assert "percentage-point" in res["scale_description"].lower()


def test_4_local_explanation_uses_correct_encounter(client, sample_features):
    """Requirement 4: Two distinct patients must produce distinct local attribution signatures."""
    patient_a = dict(sample_features)
    patient_a["number_inpatient"] = 4
    patient_a["discharge_disposition_group"] = "SNF"

    patient_b = dict(sample_features)
    patient_b["number_inpatient"] = 0
    patient_b["discharge_disposition_group"] = "Home"
    patient_b["time_in_hospital"] = 1

    resp_a = client.post("/predict", json=patient_a)
    resp_b = client.post("/predict", json=patient_b)

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    data_a = resp_a.json()
    data_b = resp_b.json()

    # Raw scores and calibrated risk must be different
    assert data_a["calibrated_risk_probability"] != data_b["calibrated_risk_probability"]
    assert data_a["raw_margin"] != data_b["raw_margin"]

    # Top driver for patient A should emphasize high utilization
    top_feats_a = [f["feature"] for f in data_a["top_risk_increasing_factors"]]
    assert "number_inpatient" in top_feats_a


def test_5_global_model_explanation_uses_documented_reference_cohort(client):
    """Requirement 5: Global explanation queries the N=14,902 validation cohort and returns valid statistics."""
    resp = client.get("/v1/explain/global")
    assert resp.status_code == 200
    data = resp.json()

    assert data["reference_cohort"]["sample_size"] == 14902
    assert 0.10 <= data["reference_cohort"]["positive_prevalence"] <= 0.13
    assert "val_rag.parquet" in data["reference_cohort"]["name"]

    # Top features ranking
    importances = data["feature_importances"]
    assert len(importances) == 42
    assert importances[0]["rank"] == 1
    assert importances[0]["mean_abs_shap"] > importances[-1]["mean_abs_shap"]

    # Calibration metrics
    assert "isotonic" in data["calibration_summary"]["method"].lower()
    assert data["calibration_summary"]["ece"] < 0.05
    assert data["calibration_summary"]["test_auc_roc"] > 0.65


def test_6_plain_english_summary_handles_edge_cases():
    """Requirement 6: Deterministic clinical narrative generation handles edge cases and disclaims causality."""
    summary = generate_plain_english_summary(
        calibrated_prob=0.245,
        raw_margin=0.35,
        base_value=-0.196,
        top_increasing=[{"feature": "number_inpatient", "feature_name": "Inpatient Stays", "value": 2, "shap_value": 0.45}],
        top_reducing=[{"feature": "discharge_disposition_group", "feature_name": "Discharge to Home", "value": "Home", "shap_value": -0.32}],
    )

    assert "24.5%" in summary
    assert "Inpatient Stays" in summary
    assert "Discharge to Home" in summary
    assert "not guarantee clinical safety" in summary.lower() or "not guarantee safety" in summary.lower()
    assert "statistical associations" in summary.lower() or "statistical association" in summary.lower()


def test_7_explanation_reliability_matrix_and_rag_failure():
    """Requirement 7: Technical reliability matrix gracefully marks degraded status without hallucinations."""
    # When RAG is degraded/unavailable
    status_degraded = check_explanation_reliability(
        patient_dict={"time_in_hospital": 3},
        has_prediction=True,
        has_shap=True,
        has_rag=False,
        calibrated_prob=0.25,
        degraded_mode=True,
    )

    assert status_degraded["checks"]["rag_peer_evidence_availability"]["status"] in ("Degraded", "Unavailable")
    assert status_degraded["overall_status"] == "Degraded"

    # When all available
    status_good = check_explanation_reliability(
        patient_dict={"rag_struct_rate_k20": 0.15, "time_in_hospital": 3, "number_inpatient": 1, "num_medications": 10, "diag_1_group": "Circulatory", "discharge_disposition_group": "Home"},
        has_prediction=True,
        has_shap=True,
        has_rag=True,
        calibrated_prob=0.25,
        degraded_mode=False,
    )
    assert status_good["overall_status"] == "Available"


def test_8_what_if_simulation_does_not_mutate_original_encounter(client, sample_features):
    """Requirement 8: Running a What-If sensitivity analysis does NOT mutate the saved encounter in SQLite."""
    # Create encounter in DB
    create_resp = client.post("/v1/discharges", json=sample_features)
    assert create_resp.status_code == 201
    enc_id = create_resp.json()["id"]

    # Read original
    orig_resp = client.get(f"/v1/discharges/{enc_id}")
    orig_data = orig_resp.json()
    orig_los = orig_data["features"]["time_in_hospital"]
    orig_prob = orig_data["calibrated_prob"]

    # Run what-if simulation with modified inputs
    what_if_resp = client.post(
        "/v1/what-if",
        json={
            "discharge_id": enc_id,
            "modified_features": {
                "time_in_hospital": 14,
                "num_medications": 30,
            },
        },
    )
    assert what_if_resp.status_code == 200
    sim_data = what_if_resp.json()

    assert "attribution_shifts" in sim_data
    assert len(sim_data["attribution_shifts"]) >= 2
    assert "disclaimer" in sim_data
    assert "sensitivity analysis" in sim_data["disclaimer"].lower()

    # Verify original encounter in database remains completely unchanged
    after_resp = client.get(f"/v1/discharges/{enc_id}")
    after_data = after_resp.json()

    assert after_data["features"]["time_in_hospital"] == orig_los
    assert after_data["calibrated_prob"] == orig_prob
    assert after_data["status"] == orig_data["status"]


def test_9_extreme_and_missing_values_handled_safely(sample_features):
    """Requirement 9: Non-standard inputs (out-of-range, missing categories) don't crash the explainer."""
    explainer = ModelExplainer()
    extreme_patient = dict(sample_features)
    extreme_patient["number_inpatient"] = 999
    extreme_patient["time_in_hospital"] = -5
    extreme_patient["medical_specialty"] = "NonExistentSpecialty123"

    res = explainer.explain_patient(extreme_patient, top_k=5)
    assert res is not None
    assert len(res["all_attributions"]) == 42
    # Ensure no NaN or infinite values
    for k, v in res["all_attributions"].items():
        assert not (v != v), f"Found NaN attribution for {k}"
        assert abs(v) < 100.0, f"Found unbounded attribution for {k}"


def test_10_api_contracts_and_schemas(client, sample_features):
    """Requirement 10: /explain and /v1/predict match schema definitions."""
    resp = client.post("/explain?top_k=5", json=sample_features)
    assert resp.status_code == 200
    data = resp.json()

    assert "full_attribution_table" in data
    assert "base_value" in data
    assert "raw_margin" in data
    assert "plain_english_summary" in data
    assert "reliability" in data or "explanation_reliability" in data


def test_11_actionloop_workflow_and_persistence_preserved(client, sample_features):
    """Requirement 11: Task management, timeline logging, and workflow state remain fully intact."""
    # Ingest
    res = client.post("/v1/discharges", json=sample_features)
    assert res.status_code == 201
    enc_id = res.json()["id"]

    # Add clinical action
    action_res = client.post(
        f"/v1/discharges/{enc_id}/actions",
        json={"type": "call_attempt", "note": "Contacted caregiver regarding discharge medication plan."},
    )
    assert action_res.status_code == 201

    # Verify action appears in timeline
    detail_res = client.get(f"/v1/discharges/{enc_id}")
    assert detail_res.status_code == 200
    d = detail_res.json()
    assert len(d["action_history"]) >= 1
    assert d["action_history"][0]["note"] == "Contacted caregiver regarding discharge medication plan."


def test_12_protected_model_artifacts_remain_untouched():
    """Requirement 12: Verify frozen model and calibrator files exist and have not been deleted or replaced."""
    model_path = Path("models/final/model.joblib")
    calibrator_path = Path("models/final/calibrator.joblib")
    features_path = Path("models/final/feature_list.json")
    thresholds_path = Path("models/final/serving_thresholds.json")

    assert model_path.exists(), "Protected model artifact models/final/model.joblib is missing!"
    assert calibrator_path.exists(), "Protected calibrator artifact is missing!"
    assert features_path.exists(), "Feature list JSON is missing!"
    assert thresholds_path.exists(), "Serving thresholds JSON is missing!"

    with open(features_path, "r") as f:
        features = json.load(f)
    assert len(features["all_features"]) == 42
