"""Unit tests for deterministic care-transition decision support interventions."""

from fastapi.testclient import TestClient
import pytest

from api.interventions import DISCLAIMER_TEXT, generate_suggested_interventions
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_interventions_disclaimer_always_present():
    """Verify that every generated intervention output contains the mandatory legal disclaimer."""
    result = generate_suggested_interventions(
        patient_data={"number_inpatient": 0, "change": "No"},
        risk_tier="Low",
    )
    assert result["disclaimer"] == DISCLAIMER_TEXT
    assert "Decision-support suggestions, not clinical advice" in result["disclaimer"]
    assert len(result["suggestions"]) > 0


def test_inpatient_utilization_triggers_rapid_outreach():
    """Verify that multiple prior inpatient admissions trigger 48h telephonic outreach and PCP visit."""
    patient = {
        "number_inpatient": 3,
        "total_prior_visits": 4,
        "change": "No",
        "num_medications": 10,
    }
    result = generate_suggested_interventions(patient, risk_tier="High")
    ids = [s["id"] for s in result["suggestions"]]

    assert "rapid_telephonic_followup" in ids
    assert "pcp_clinic_handoff" in ids

    rapid_sugg = next(s for s in result["suggestions"] if s["id"] == "rapid_telephonic_followup")
    assert "Prior Inpatient Admissions" in rapid_sugg["trigger_driver"]
    assert rapid_sugg["priority"] == "High"
    assert "48 hours" in rapid_sugg["target_timeline"]


def test_medication_change_triggers_pharmacist_review():
    """Verify that active medication regimen changes trigger clinical pharmacist medication review."""
    patient = {
        "change": "Ch",
        "insulin": "Up",
        "n_dose_changes": 1,
        "number_inpatient": 0,
    }
    result = generate_suggested_interventions(
        patient,
        risk_tier="High",
        top_risk_drivers=[{"feature": "change", "explanation": "Medication change raises risk"}],
    )
    ids = [s["id"] for s in result["suggestions"]]

    assert "pharmacist_med_recon" in ids
    recon_sugg = next(s for s in result["suggestions"] if s["id"] == "pharmacist_med_recon")
    assert "Inpatient Diabetes Regimen Change" in recon_sugg["trigger_driver"]
    assert recon_sugg["priority"] == "High"


def test_polypharmacy_triggers_simplification():
    """Verify that excessive medication count (>=16) triggers regimen simplification."""
    patient = {
        "num_medications": 22,
        "polypharmacy": 1,
        "number_inpatient": 0,
        "change": "No",
    }
    result = generate_suggested_interventions(patient, risk_tier="Medium")
    ids = [s["id"] for s in result["suggestions"]]

    assert "polypharmacy_review" in ids
    poly_sugg = next(s for s in result["suggestions"] if s["id"] == "polypharmacy_review")
    assert "High Medication Burden" in poly_sugg["trigger_driver"]


def test_home_health_triggers_agency_handoff():
    """Verify that discharge to home health triggers warm agency handoff confirmation."""
    patient = {
        "discharge_disposition_group": "Home Health",
        "number_inpatient": 1,
        "change": "No",
    }
    result = generate_suggested_interventions(patient, risk_tier="High")
    ids = [s["id"] for s in result["suggestions"]]

    assert "home_health_warm_handoff" in ids
    hh_sugg = next(s for s in result["suggestions"] if s["id"] == "home_health_warm_handoff")
    assert "Discharge Disposition" in hh_sugg["trigger_driver"]
    assert hh_sugg["priority"] == "High"


def test_predict_endpoint_includes_suggested_interventions(client):
    """Integration test: confirm POST /predict returns suggested_interventions in its payload."""
    payload = {
        "time_in_hospital": 6,
        "number_inpatient": 2,
        "number_emergency": 1,
        "num_medications": 18,
        "diag_1_group": "Circulatory",
        "A1Cresult": ">8",
        "change": "Ch",
        "insulin": "Up",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "suggested_interventions" in data
    interventions = data["suggested_interventions"]
    assert interventions["disclaimer"] == "Decision-support suggestions, not clinical advice"
    assert interventions["suggestions_count"] >= 2
    assert len(interventions["suggestions"]) >= 2

    # Check structure of items
    first = interventions["suggestions"][0]
    assert "id" in first
    assert "action" in first
    assert "category" in first
    assert "priority" in first
    assert "trigger_driver" in first
    assert "rationale" in first
    assert "target_timeline" in first
