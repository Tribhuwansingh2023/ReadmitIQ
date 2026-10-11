import io
import json

import pytest
from fastapi.testclient import TestClient

from api.database import get_db, init_db
from api.main import app
from api.service import get_service
from src.config import MODELS_DIR

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Run tests with an isolated test database in tmp_path."""
    db_file = tmp_path / "test_readmitiq.db"
    monkeypatch.setenv("READMITIQ_DB_PATH", str(db_file))
    init_db(db_file)
    yield


@pytest.fixture
def sample_encounter():
    return {
        "age": "[60-70)",
        "gender": "Female",
        "race": "Caucasian",
        "time_in_hospital": 4,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 12,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 7,
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "discharge_disposition_group": "Home",
        "medical_specialty": "InternalMedicine",
        "payer_code": "Medicare",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Other",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "diabetesMed": "Yes",
        "change": "No",
        "insulin": "No",
        "metformin": "Steady",
    }


def test_model_version_header_present_on_all_responses(sample_encounter):
    """Verify X-Model-Version header is present on responses."""
    resp = client.get("/health")
    assert resp.headers.get("X-Model-Version") == "1.0.0 (LightGBM + Structured RAG)"

    resp2 = client.get("/v1/worklist")
    assert resp2.headers.get("X-Model-Version") == "1.0.0 (LightGBM + Structured RAG)"


def test_scoring_determinism_and_tie_breaks(sample_encounter):
    """Verify identical inputs produce identical scores and worklist tie-breaks by ID ASC."""
    svc = get_service()

    # Determinism
    score1 = svc.score_batch([sample_encounter])[0]
    score2 = svc.score_batch([sample_encounter])[0]
    assert score1["raw_score"] == score2["raw_score"]
    assert score1["calibrated_prob"] == score2["calibrated_prob"]
    assert score1["tier"] == score2["tier"]

    # Ingest two identical encounters
    resp1 = client.post("/v1/discharges", json=sample_encounter)
    resp2 = client.post("/v1/discharges", json=sample_encounter)
    id1 = resp1.json()["id"]
    id2 = resp2.json()["id"]
    assert id1 < id2

    # Worklist ordering with identical raw scores must tie-break by id ASC
    wl_resp = client.get("/v1/worklist")
    items = wl_resp.json()["items"]
    ids = [item["id"] for item in items if item["id"] in (id1, id2)]
    assert ids == [id1, id2]


def test_feed_scoring_and_predict_endpoint_identity(sample_encounter):
    """Verify the simulated feed path and /predict yield identical model predictions."""
    svc = get_service()

    predict_res = svc.predict(dict(sample_encounter))
    batch_res = svc.score_batch([dict(sample_encounter)])[0]

    # Both must match calibrated probability
    assert predict_res["calibrated_risk_probability"] == pytest.approx(
        batch_res["calibrated_prob"], abs=1e-3
    )
    assert predict_res["risk_tier"] == batch_res["tier"]


def test_db_round_trip_and_features_persistence(sample_encounter):
    """Verify encounter persistence to SQLite and detail retrieval with on-demand TreeSHAP."""
    post_resp = client.post(
        "/v1/discharges",
        json={**sample_encounter, "external_ref": "REF-TEST-001", "source": "api"},
    )
    assert post_resp.status_code == 201
    discharge_id = post_resp.json()["id"]

    # Retrieve encounter details
    get_resp = client.get(f"/v1/discharges/{discharge_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()

    assert data["id"] == discharge_id
    assert data["external_ref"] == "REF-TEST-001"
    assert data["source"] == "api"
    assert data["status"] == "new"
    assert "shap_explanation" in data
    assert "top_risk_increasing" in data["shap_explanation"]
    assert "suggested_interventions" in data
    assert len(data["suggested_interventions"]["suggestions"]) > 0


def test_validation_errors_single_and_batch():
    """Verify clear per-row errors on invalid schemas."""
    # Single invalid record -> 422
    bad_single = {"time_in_hospital": 0, "num_medications": -1}
    resp_single = client.post("/v1/discharges", json=bad_single)
    assert resp_single.status_code == 422

    # Batch with mixed valid and invalid records
    good_rec = {
        "time_in_hospital": 3,
        "num_medications": 10,
        "num_lab_procedures": 30,
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "discharge_disposition_group": "Home",
        "medical_specialty": "InternalMedicine",
        "payer_code": "Medicare",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Other",
        "A1Cresult": "None",
        "max_glu_serum": "None",
        "diabetesMed": "No",
        "change": "No",
        "insulin": "No",
        "external_ref": "VALID-1",
    }
    batch_payload = [
        good_rec,
        {"time_in_hospital": 999, "num_medications": -5, "external_ref": "INVALID-1"},
    ]
    resp_batch = client.post("/v1/discharges", json=batch_payload)
    assert resp_batch.status_code == 201
    bdata = resp_batch.json()
    assert bdata["accepted_count"] == 1
    assert bdata["rejected_count"] == 1
    assert len(bdata["errors"]) == 1
    assert bdata["errors"][0]["external_ref"] == "INVALID-1"


def test_audit_entries_on_every_mutation(sample_encounter):
    """Verify audit log records every workflow mutation."""
    post_resp = client.post("/v1/discharges", json=sample_encounter)
    discharge_id = post_resp.json()["id"]

    # 1. Update status and assignee
    patch_resp = client.patch(
        f"/v1/discharges/{discharge_id}",
        json={"status": "contacted", "assignee": "Nurse Sarah", "actor": "Nurse Sarah"},
    )
    assert patch_resp.status_code == 200

    # 2. Add clinical action
    action_resp = client.post(
        f"/v1/discharges/{discharge_id}/actions",
        json={
            "type": "phone_call",
            "note": "Spoke with patient regarding post-discharge medication schedule.",
            "actor": "Nurse Sarah",
        },
    )
    assert action_resp.status_code == 201

    # 3. Record outcome
    outcome_resp = client.post(
        f"/v1/discharges/{discharge_id}/outcome",
        json={"outcome_readmitted_30d": 1, "actor": "Quality Team"},
    )
    assert outcome_resp.status_code == 200

    # Verify audit log in SQLite
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT event, actor FROM audit_log ORDER BY id ASC"
        )
        events = [(r["event"], r["actor"]) for r in cursor.fetchall()]

    event_names = [e[0] for e in events]
    assert "discharge_created" in event_names
    assert "discharge_updated" in event_names
    assert "action_added" in event_names
    assert "outcome_recorded" in event_names


def test_clinician_override_requires_reason(sample_encounter):
    """Verify tier override strictly rejects missing clinical reason."""
    post_resp = client.post("/v1/discharges", json=sample_encounter)
    d_id = post_resp.json()["id"]

    # Missing reason -> 400 Bad Request
    fail_resp = client.patch(
        f"/v1/discharges/{d_id}",
        json={"clinician_override_tier": "Critical", "override_reason": ""},
    )
    assert fail_resp.status_code == 400

    # With clinical reason -> succeeds
    ok_resp = client.patch(
        f"/v1/discharges/{d_id}",
        json={
            "clinician_override_tier": "Critical",
            "override_reason": "Patient lives alone with poor social support and mobility issues.",
        },
    )
    assert ok_resp.status_code == 200


def test_sealed_truth_not_reachable_before_time_machine():
    """Verify sealed feed_truth label is hidden until time-machine release."""
    from api.feed import get_feed_engine

    engine = get_feed_engine()
    # Emit single encounter
    emitted = engine.emit_single_encounter()
    assert emitted is not None
    d_id = emitted["discharge_id"]

    # 1. API GET /v1/discharges/{id} must show outcome as None
    detail_resp = client.get(f"/v1/discharges/{d_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["outcome_readmitted_30d"] is None

    # 2. Worklist must show outcome as None
    wl_resp = client.get("/v1/worklist")
    found = [item for item in wl_resp.json()["items"] if item["id"] == d_id][0]
    assert found["outcome_readmitted_30d"] is None

    # 3. Advance time machine by 35 days (> 30 days)
    tm_resp = client.post("/v1/feed/time-machine", json={"days": 35})
    assert tm_resp.status_code == 200
    assert tm_resp.json()["outcomes_released"] >= 1

    # 4. Now outcome is released and visible
    detail_after = client.get(f"/v1/discharges/{d_id}")
    assert detail_after.json()["outcome_readmitted_30d"] in (0, 1)


def test_monitoring_guards_under_small_n(sample_encounter):
    """Verify GET /v1/monitoring guards under small sample size (<200 outcomes)."""
    # Create single discharge with outcome
    post_resp = client.post("/v1/discharges", json=sample_encounter)
    d_id = post_resp.json()["id"]
    client.post(f"/v1/discharges/{d_id}/outcome", json={"outcome_readmitted_30d": 1})

    mon_resp = client.get("/v1/monitoring")
    assert mon_resp.status_code == 200
    mdata = mon_resp.json()

    assert mdata["outcomes_evaluation"]["insufficient_outcomes"] is True
    assert "Insufficient outcome data" in mdata["outcomes_evaluation"]["status_message"]
    assert "auc_roc" not in mdata["outcomes_evaluation"]


def test_threshold_tiers_match_serving_thresholds():
    """Verify tier assignments match serving_thresholds.json cutoffs."""
    svc = get_service()

    with open(MODELS_DIR / "final" / "serving_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    cutoffs = thresholds["raw_score_cutoffs"]

    assert svc.assign_tier(cutoffs["top_5pct"] + 0.001) == "Critical"
    assert svc.assign_tier(cutoffs["top_10pct"] + 0.001) == "High"
    assert svc.assign_tier(cutoffs["top_20pct"] + 0.001) == "Medium"
    assert svc.assign_tier(cutoffs["top_20pct"] - 0.010) == "Low"


def test_csv_template_and_upload():
    """Verify CSV template download and CSV upload parsing."""
    temp_resp = client.get("/v1/templates/discharge.csv")
    assert temp_resp.status_code == 200
    assert "text/csv" in temp_resp.headers["Content-Type"]
    csv_text = temp_resp.content.decode("utf-8")
    assert "time_in_hospital" in csv_text
    assert "A1Cresult" in csv_text

    # Upload the template back to the upload endpoint
    files = {"file": ("test_upload.csv", io.BytesIO(temp_resp.content), "text/csv")}
    up_resp = client.post("/v1/discharges/upload", files=files)
    assert up_resp.status_code == 200
    up_data = up_resp.json()
    assert up_data["total_rows_received"] == 3
    assert up_data["accepted_count"] == 3
    assert up_data["rejected_count"] == 0
