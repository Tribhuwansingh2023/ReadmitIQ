"""Comprehensive integration tests for ReadmitIQ ActionLoop and signature differentiators.

Validates:
- ActionLoop Stage 1: Risk Intelligence & Scoring Determinism
- ActionLoop Stage 2: Precedent Evidence & RAG Context
- ActionLoop Stage 3: Care Transition Action Plan Tasks & Status Lifecycle
- ActionLoop Stage 4: Capacity-Aware Priority Triage & Deterministic Tie-breaking
- ActionLoop Stage 5: Closed-Loop Follow-up & Chronological Audit Timeline
- Differentiator 1: What-If Sensitivity Risk Explorer
- Differentiator 2: Trust & Safety Monitoring with Small-n Statistical Guardrails
- Data Quality Ingestion & Rejection Logging
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.database import get_db, init_db
from api.main import app
from api.service import get_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Run tests with an isolated test database in tmp_path."""
    db_file = tmp_path / "test_actionloop.db"
    monkeypatch.setenv("READMITIQ_DB_PATH", str(db_file))
    init_db(db_file)
    yield


@pytest.fixture
def high_risk_encounter():
    """Sample clinical encounter with multi-morbidity and high inpatient utilization."""
    return {
        "age": "[70-80)",
        "gender": "Female",
        "race": "Caucasian",
        "time_in_hospital": 8,
        "num_lab_procedures": 55,
        "num_procedures": 2,
        "num_medications": 22,
        "number_outpatient": 1,
        "number_emergency": 2,
        "number_inpatient": 3,
        "number_diagnoses": 9,
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "discharge_disposition_group": "Home",
        "medical_specialty": "InternalMedicine",
        "payer_code": "Medicare",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Respiratory",
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "diabetesMed": "Yes",
        "change": "Ch",
        "insulin": "Steady",
        "metformin": "No",
    }


def test_actionloop_stage1_risk_intelligence(high_risk_encounter):
    """Stage 1: Verify predict returns raw score, calibrated probability, tier, and SHAP explanation."""
    resp = client.post("/v1/predict", json=high_risk_encounter)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "raw_score" in data
    assert "calibrated_prob" in data or "calibrated_risk_probability" in data
    assert data["tier"] in ["Critical", "High", "Medium", "Low"]
    assert "shap_explanation" in data

    # Strict floating-point determinism check
    resp2 = client.post("/v1/predict", json=high_risk_encounter)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data["raw_score"] == data2["raw_score"]
    assert data["tier"] == data2["tier"]


def test_actionloop_stage2_precedent_evidence(high_risk_encounter):
    """Stage 2: Verify intake creates encounter with RAG precedent retrieval."""
    resp = client.post(
        "/v1/discharges",
        json={**high_risk_encounter, "external_ref": "ENC-ACT-001", "source": "api"},
    )
    assert resp.status_code in [200, 201], resp.text
    created = resp.json()
    discharge_id = created["id"]

    # Fetch brief detail
    detail_resp = client.get(f"/v1/discharges/{discharge_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()

    assert "retrieval" in detail
    assert "top_neighbors" in detail["retrieval"]
    neighbors = detail["retrieval"]["top_neighbors"]
    assert isinstance(neighbors, list)
    if len(neighbors) > 0:
        neighbor = neighbors[0]
        assert "similarity_score" in neighbor or "similarity" in neighbor
        assert "age" in neighbor or "features" in neighbor
        assert "readmitted_30d" in neighbor or "outcome_readmitted_30d" in neighbor


def test_actionloop_stage3_action_plan_task_lifecycle(high_risk_encounter):
    """Stage 3: Verify creation, listing, and state transition lifecycle of Action Plan tasks."""
    # Create encounter
    intake_resp = client.post(
        "/v1/discharges",
        json={**high_risk_encounter, "external_ref": "ENC-TASK-001"},
    )
    assert intake_resp.status_code in [200, 201]
    discharge_id = intake_resp.json()["id"]

    # Initial tasks query
    tasks_resp = client.get(f"/v1/discharges/{discharge_id}/tasks")
    assert tasks_resp.status_code == 200

    # Add custom high-priority task
    new_task_payload = {
        "title": "Urgent Post-Discharge Cardiology Telehealth Follow-up",
        "category": "clinical_followup",
        "priority": "High",
        "owner": "Nurse Coordinator Jane",
        "rationale": "High circulatory readmission risk with prior inpatient stays",
    }
    create_resp = client.post(f"/v1/discharges/{discharge_id}/tasks", json=new_task_payload)
    assert create_resp.status_code in [200, 201]
    task_created = create_resp.json()["task"]
    task_id = task_created["id"]
    assert task_created["status"] == "pending"
    assert task_created["priority"] == "High"

    # Transition to in_progress
    update_1 = client.patch(
        f"/v1/discharges/{discharge_id}/tasks/{task_id}",
        json={"status": "in_progress"},
    )
    assert update_1.status_code == 200
    assert update_1.json()["task"]["status"] == "in_progress"

    # Complete task with note
    update_2 = client.patch(
        f"/v1/discharges/{discharge_id}/tasks/{task_id}",
        json={
            "status": "completed",
            "actor": "Nurse Coordinator Jane",
            "completed_by": "Nurse Coordinator Jane",
            "completion_note": "Telehealth appointment confirmed for 48h post-discharge.",
        },
    )
    assert update_2.status_code == 200
    assert update_2.json()["task"]["status"] == "completed"
    assert update_2.json()["task"]["completed_by"] == "Nurse Coordinator Jane"


def test_actionloop_stage4_capacity_aware_triage(high_risk_encounter):
    """Stage 4: Verify deterministic capacity triage ranking and queue context."""
    # Create low-risk and high-risk encounters
    low_risk = dict(high_risk_encounter)
    low_risk["time_in_hospital"] = 1
    low_risk["num_medications"] = 5
    low_risk["number_inpatient"] = 0
    low_risk["number_emergency"] = 0

    resp_high = client.post(
        "/v1/discharges", json={**high_risk_encounter, "external_ref": "ENC-HIGH-001"}
    )
    resp_low = client.post(
        "/v1/discharges", json={**low_risk, "external_ref": "ENC-LOW-001"}
    )
    assert resp_high.status_code in [200, 201]
    assert resp_low.status_code in [200, 201]

    high_id = resp_high.json()["id"]
    low_id = resp_low.json()["id"]

    # Verify worklist order: higher raw score ranks ahead of lower raw score
    worklist_resp = client.get("/v1/worklist")
    assert worklist_resp.status_code == 200
    items = worklist_resp.json()["items"]
    ids_in_order = [item["id"] for item in items]

    assert ids_in_order.index(high_id) < ids_in_order.index(low_id)

    # Verify detail response includes capacity_triage structure
    detail_resp = client.get(f"/v1/discharges/{high_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert "capacity_triage" in detail
    assert "priority_rank" in detail["capacity_triage"]
    assert "is_in_capacity" in detail["capacity_triage"]


def test_actionloop_stage5_closed_loop_followup_and_timeline(high_risk_encounter):
    """Stage 5: Verify care actions, outcome recording, and chronological timeline synthesis."""
    # Create encounter
    intake_resp = client.post(
        "/v1/discharges", json={**high_risk_encounter, "external_ref": "ENC-TIMELINE-001"}
    )
    assert intake_resp.status_code in [200, 201]
    discharge_id = intake_resp.json()["id"]

    # Record outreach action
    action_resp = client.post(
        f"/v1/discharges/{discharge_id}/actions",
        json={
            "type": "phone_call",
            "note": "Spoke with patient; verified prescription pickup and home health aid.",
            "actor": "Case Manager Alex",
        },
    )
    assert action_resp.status_code in [200, 201]

    # Record 30-day sealed outcome
    outcome_resp = client.post(
        f"/v1/discharges/{discharge_id}/outcome",
        json={"outcome_readmitted_30d": 0, "notes": "No 30-day readmission observed."},
    )
    assert outcome_resp.status_code in [200, 201]
    assert outcome_resp.json()["outcome_readmitted_30d"] == 0

    # Fetch brief and check chronological timeline
    detail_resp = client.get(f"/v1/discharges/{discharge_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()

    assert "timeline" in detail
    timeline = detail["timeline"]
    assert len(timeline) >= 3  # Intake, Action, Outcome

    event_types = [event["type"] for event in timeline]
    assert "intake" in event_types
    assert "action" in event_types
    assert "outcome" in event_types
    assert detail["outcome_readmitted_30d"] == 0


def test_differentiator_what_if_risk_explorer(high_risk_encounter):
    """Differentiator 1: Test what-if risk sensitivity calculation and directional monotonicity."""
    # Intake patient
    intake_resp = client.post(
        "/v1/discharges", json={**high_risk_encounter, "external_ref": "ENC-WHATIF-001"}
    )
    assert intake_resp.status_code in [200, 201]
    discharge_id = intake_resp.json()["id"]

    # Explore risk reduction by reducing length of stay and prior admissions
    modified_features = dict(high_risk_encounter)
    modified_features["time_in_hospital"] = 2
    modified_features["number_inpatient"] = 0
    modified_features["number_emergency"] = 0

    whatif_resp = client.post(
        "/v1/what-if",
        json={"discharge_id": discharge_id, "modified_features": modified_features},
    )
    assert whatif_resp.status_code == 200
    res = whatif_resp.json()

    assert "baseline" in res
    assert "modified" in res
    assert "delta" in res

    # Monotonicity check: decreasing acute severity features should reduce or preserve raw score
    assert res["modified"]["raw_score"] <= res["baseline"]["raw_score"]
    assert res["delta"]["probability_change"] <= 0.05


def test_differentiator_trust_safety_monitoring_and_small_n_guard():
    """Differentiator 2: Verify live monitoring endpoint with small-n statistical guardrail."""
    mon_resp = client.get("/v1/monitoring")
    assert mon_resp.status_code == 200
    mon = mon_resp.json()

    assert "counts" in mon
    assert "tier_mix" in mon
    assert "score_drift" in mon
    assert "rag_health" in mon
    assert "outcomes_evaluation" in mon

    # Small-n guard verification
    eval_info = mon["outcomes_evaluation"]
    assert eval_info["minimum_required_outcomes"] == 200
    assert eval_info["minimum_required_positives"] == 20
    # In an empty or small test DB, insufficient_outcomes must be True
    assert eval_info["insufficient_outcomes"] is True
    assert "insufficient outcome data" in eval_info["status_message"].lower()

    # RAG vector store health
    assert mon["rag_health"]["status"] in ["Operational", "Unavailable"]
    assert mon["rag_health"]["embedding_dimension"] == 64


def test_data_quality_rejection_logging_and_resilience():
    """Data Quality: Verify invalid inputs are handled cleanly with descriptive rejection."""
    # Send invalid payload violating ge=1 schema constraints
    bad_resp = client.post("/v1/predict", json={"time_in_hospital": 0, "num_medications": -5})
    assert bad_resp.status_code == 422  # FastAPI schema validation rejection

    # Verify database query still functions normally
    worklist_resp = client.get("/v1/worklist")
    assert worklist_resp.status_code == 200
