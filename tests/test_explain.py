"""Comprehensive tests for ModelExplainer portability, dual backends, and error safety."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.explain import (
    SHAP_AVAILABLE,
    ExplainerError,
    ModelExplainer,
)


@pytest.fixture
def sample_valid_patient() -> Dict[str, Any]:
    return {
        "number_inpatient": 2,
        "number_emergency": 0,
        "number_outpatient": 1,
        "total_prior_visits": 3,
        "time_in_hospital": 4,
        "num_medications": 14,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "number_diagnoses": 8,
        "age_num": 65,
        "polypharmacy": 0,
        "n_active_drugs": 2,
        "n_dose_changes": 0,
        "labs_per_day": 11.25,
        "meds_per_day": 3.5,
        "a1c_tested": 0,
        "rag_struct_rate_k20": 0.18,
        "discharge_disposition_group": "Home",
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Other",
        "A1Cresult": "None",
        "max_glu_serum": "None",
        "diabetesMed": "Yes",
        "change": "No",
        "insulin": "Steady",
        "metformin": "Steady",
        "repaglinide": "No",
        "nateglinide": "No",
        "glimepiride": "No",
        "glipizide": "No",
        "glyburide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "glyburide-metformin": "No",
        "race": "Caucasian",
        "gender": "Female",
        "medical_specialty": "InternalMedicine",
        "payer_code": "Medicare",
    }


def test_explainer_auto_operation(sample_valid_patient):
    """Verify ModelExplainer initializes under auto-detection and generates valid explanations."""
    explainer = ModelExplainer()
    assert explainer.backend in ("shap", "lightgbm_native")
    result = explainer.explain_patient(sample_valid_patient, top_k=3)

    assert "top_risk_increasing" in result
    assert "top_risk_reducing" in result
    assert "all_attributions" in result
    assert len(result["all_attributions"]) == 42
    assert len(result["top_risk_increasing"]) <= 3
    assert len(result["top_risk_reducing"]) <= 3

    # Verify structure and format of each factor
    for factor in result["top_risk_increasing"]:
        assert factor["shap_value"] > 0
        assert "raises risk" in factor["explanation"]
    for factor in result["top_risk_reducing"]:
        assert factor["shap_value"] < 0
        assert "lowers risk" in factor["explanation"]


def test_explainer_forced_native_fallback(sample_valid_patient):
    """Verify forced LightGBM native TreeSHAP backend operates without SHAP dependency."""
    explainer = ModelExplainer(force_backend="lightgbm_native")
    assert explainer.backend == "lightgbm_native"
    assert explainer.explainer is None

    result = explainer.explain_patient(sample_valid_patient, top_k=5)
    assert result["backend"] == "lightgbm_native"
    assert len(result["all_attributions"]) == 42
    assert len(result["top_risk_increasing"]) <= 5

    # Check that highest positive risk driver is properly formatted
    if result["top_risk_increasing"]:
        top_driver = result["top_risk_increasing"][0]
        assert isinstance(top_driver["shap_value"], float)
        assert len(top_driver["explanation"]) > 5


def test_explainer_backend_attribution_parity(sample_valid_patient):
    """Verify SHAP and native LightGBM TreeSHAP produce identical feature attributions."""
    if not SHAP_AVAILABLE:
        pytest.skip("SHAP is not available in current environment; skipping parity comparison.")

    explainer_shap = ModelExplainer(force_backend="shap")
    explainer_native = ModelExplainer(force_backend="lightgbm_native")

    res_shap = explainer_shap.explain_patient(sample_valid_patient, top_k=5)
    res_native = explainer_native.explain_patient(sample_valid_patient, top_k=5)

    attrs_shap = res_shap["all_attributions"]
    attrs_native = res_native["all_attributions"]

    assert set(attrs_shap.keys()) == set(attrs_native.keys())
    for feat in attrs_shap:
        assert np.isclose(attrs_shap[feat], attrs_native[feat], atol=1e-3), (
            f"Feature {feat} attribution mismatch: SHAP={attrs_shap[feat]}, Native={attrs_native[feat]}"
        )


def test_explainer_malformed_inputs():
    """Verify ModelExplainer rejects invalid or malformed patient data with clean ExplainerError."""
    explainer = ModelExplainer(force_backend="lightgbm_native")

    # None input
    with pytest.raises(ExplainerError, match="cannot be None"):
        explainer.explain_patient(None)

    # String input
    with pytest.raises(ExplainerError, match="Unsupported patient_data type"):
        explainer.explain_patient("invalid_string_payload")

    # Empty dictionary gracefully defaults missing fields without unhandled crash
    result = explainer.explain_patient({})
    assert "top_risk_increasing" in result
    assert len(result["all_attributions"]) == 42


def test_explainer_initialization_failures():
    """Verify ModelExplainer raises ExplainerError when artifact files do not exist."""
    fake_path = Path("non_existent_model.joblib")
    with pytest.raises(ExplainerError, match="Model file not found"):
        ModelExplainer(model_path=fake_path)

    fake_meta = Path("non_existent_meta.json")
    with pytest.raises(ExplainerError, match="Feature list file not found"):
        ModelExplainer(feature_list_path=fake_meta)


def test_api_explain_endpoint_error_safety(sample_valid_patient):
    """Verify /explain endpoint handles valid input and rejects malformed inputs safely."""
    client = TestClient(app)

    # 1. Valid request
    resp = client.post("/explain", json=sample_valid_patient)
    assert resp.status_code == 200
    data = resp.json()
    assert "top_risk_increasing" in data
    assert "top_risk_reducing" in data
    assert "all_attributions" in data
    assert len(data["all_attributions"]) == 42

    # 2. Malformed request (invalid type in integer field)
    malformed_payload = dict(sample_valid_patient)
    malformed_payload["time_in_hospital"] = "not_a_number"
    resp_err = client.post("/explain", json=malformed_payload)
    assert resp_err.status_code == 422
    err_body = resp_err.json()
    assert "detail" in err_body
    # Ensure internal server path or stack traces are not leaked
    assert "Traceback" not in str(err_body)
    assert "File \"" not in str(err_body)
    assert "ValueError" not in str(err_body)


def test_service_explain_rag_failure_strict_mode(sample_valid_patient, monkeypatch):
    """Verify ModelService.explain raises ExplainerError in strict mode when RAG retrieval fails."""
    from api.service import get_service
    svc = get_service()

    patient_no_rag = dict(sample_valid_patient)
    patient_no_rag.pop("rag_struct_rate_k20", None)

    def failing_retriever(*args, **kwargs):
        raise RuntimeError("FAISS cluster connection timeout")

    monkeypatch.setattr(svc, "retrieve_similar", failing_retriever)

    # In strict mode (allow_degraded=False), must raise ExplainerError rather than inventing 0.1139
    with pytest.raises(ExplainerError, match="RAG neighborhood retrieval failed and degraded mode is disabled"):
        svc.explain(patient_no_rag, top_k=3, allow_degraded=False)


def test_service_explain_rag_failure_degraded_mode(sample_valid_patient, monkeypatch):
    """Verify ModelService.explain safely excludes peer factors and discloses degraded mode when RAG fails."""
    from api.service import get_service
    svc = get_service()

    patient_no_rag = dict(sample_valid_patient)
    patient_no_rag.pop("rag_struct_rate_k20", None)

    def failing_retriever(*args, **kwargs):
        raise RuntimeError("Simulated FAISS retrieval index corrupt")

    monkeypatch.setattr(svc, "retrieve_similar", failing_retriever)

    # In degraded mode (allow_degraded=True), succeeds with explicit disclosure
    res = svc.explain(patient_no_rag, top_k=3, allow_degraded=True)
    assert res["degraded_mode"] is True
    assert "peer-based attribution excluded" in res["degraded_reason"]

    # Verify RAG features are strictly stripped from clinical drivers
    for factor in res["top_risk_increasing"] + res["top_risk_reducing"]:
        assert factor["feature"] not in ("rag_struct_rate_k20", "rag_readmit_rate_k20")
        assert "similar historical patients" not in factor["explanation"].lower()


def test_api_explain_endpoint_degraded_mode_flow(sample_valid_patient, monkeypatch):
    """Verify /explain endpoint honors allow_degraded query flag when RAG retrieval fails."""
    from api.service import get_service
    svc = get_service()

    def failing_retriever(*args, **kwargs):
        raise RuntimeError("FAISS offline")

    monkeypatch.setattr(svc, "retrieve_similar", failing_retriever)
    client = TestClient(app)

    patient_no_rag = dict(sample_valid_patient)
    patient_no_rag["rag_struct_rate_k20"] = None

    # Strict mode defaults to False -> returns safe 422 error
    resp_strict = client.post("/explain?allow_degraded=false", json=patient_no_rag)
    assert resp_strict.status_code == 422
    err_body = resp_strict.json()
    assert "Invalid encounter input data for explainability" in err_body["detail"]
    assert "Traceback" not in str(err_body)
    assert "RuntimeError" not in str(err_body)

    # Degraded mode allowed -> returns 200 with degraded disclosure
    resp_degraded = client.post("/explain?allow_degraded=true", json=patient_no_rag)
    assert resp_degraded.status_code == 200
    data = resp_degraded.json()
    assert data["degraded_mode"] is True
    assert "peer-based attribution excluded" in data["degraded_reason"]
    for factor in data["top_risk_increasing"] + data["top_risk_reducing"]:
        assert factor["feature"] not in ("rag_struct_rate_k20", "rag_readmit_rate_k20")


def test_api_predict_and_similar_sanitized_errors():
    """Verify /predict and /similar do not leak internal Python exception strings or paths."""
    client = TestClient(app)

    # Predict with invalid field type
    resp_pred = client.post("/predict", json={"time_in_hospital": -5})
    assert resp_pred.status_code == 422
    pred_err = resp_pred.json()
    assert "Traceback" not in str(pred_err)
    assert "ValueError" not in str(pred_err)

    # Similar with invalid k parameter
    resp_sim = client.post("/similar", json={"patient_features": {}, "k": 100})
    assert resp_sim.status_code == 422
    sim_err = resp_sim.json()
    assert "Traceback" not in str(sim_err)
    assert "TypeError" not in str(sim_err)


def test_service_predict_rag_failure_strict_mode(sample_valid_patient, monkeypatch):
    """Verify ModelService.predict raises ExplainerError in strict mode when RAG fails."""
    from api.service import get_service
    svc = get_service()

    patient_no_rag = dict(sample_valid_patient)
    patient_no_rag.pop("rag_struct_rate_k20", None)

    def failing_retriever(*args, **kwargs):
        raise RuntimeError("FAISS cluster connection timeout")

    monkeypatch.setattr(svc, "retrieve_similar", failing_retriever)

    with pytest.raises(ExplainerError, match="RAG neighborhood retrieval failed and degraded mode is disabled"):
        svc.predict(patient_no_rag, allow_degraded=False)


def test_service_predict_rag_failure_degraded_mode(sample_valid_patient, monkeypatch):
    """Verify ModelService.predict safely operates in degraded mode without fabricating peers."""
    from api.service import get_service
    svc = get_service()

    patient_no_rag = dict(sample_valid_patient)
    patient_no_rag.pop("rag_struct_rate_k20", None)

    def failing_retriever(*args, **kwargs):
        raise RuntimeError("Simulated FAISS retrieval index corrupt")

    monkeypatch.setattr(svc, "retrieve_similar", failing_retriever)

    res = svc.predict(patient_no_rag, allow_degraded=True)
    assert res["degraded_mode"] is True
    assert "RAG neighborhood retrieval unavailable" in res["degraded_reason"]
    assert res["similar_patients_summary"]["degraded"] is True
    assert res["similar_patients_summary"]["k_neighbors"] == 0
    assert res["similar_patients_summary"]["top_neighbors"] == []

    # Peer factors must NOT be in clinical drivers
    for factor in res["top_risk_increasing_factors"] + res["top_risk_reducing_factors"]:
        assert factor["feature"] not in ("rag_struct_rate_k20", "rag_readmit_rate_k20")
        assert "similar historical patients" not in factor["explanation"].lower()


def test_api_predict_endpoint_degraded_mode_flow(sample_valid_patient, monkeypatch):
    """Verify /predict endpoint handles retrieval failure in strict and degraded modes."""
    from api.service import get_service
    svc = get_service()

    def failing_retriever(*args, **kwargs):
        raise RuntimeError("FAISS offline")

    monkeypatch.setattr(svc, "retrieve_similar", failing_retriever)
    client = TestClient(app)

    patient_no_rag = dict(sample_valid_patient)
    patient_no_rag["rag_struct_rate_k20"] = None

    # Strict mode defaults to False -> returns safe 422 error
    resp_strict = client.post("/predict?allow_degraded=false", json=patient_no_rag)
    assert resp_strict.status_code == 422
    err_body = resp_strict.json()
    assert "Inference processing failed due to missing retrieval context" in err_body["detail"]
    assert "Traceback" not in str(err_body)
    assert "RuntimeError" not in str(err_body)

    # Degraded mode allowed -> returns 200 with degraded disclosure
    resp_degraded = client.post("/predict?allow_degraded=true", json=patient_no_rag)
    assert resp_degraded.status_code == 200
    data = resp_degraded.json()
    assert data["degraded_mode"] is True
    assert "RAG neighborhood retrieval unavailable" in data["degraded_reason"]
    assert data["similar_patients_summary"]["k_neighbors"] == 0
    assert data["similar_patients_summary"]["top_neighbors"] == []
    for factor in data["top_risk_increasing_factors"] + data["top_risk_reducing_factors"]:
        assert factor["feature"] not in ("rag_struct_rate_k20", "rag_readmit_rate_k20")


def test_successful_rag_retrieval_flow(sample_valid_patient):
    """Confirm successful RAG retrieval preserves neighbor summaries and full factor attribution."""
    client = TestClient(app)

    resp = client.post("/predict", json=sample_valid_patient)
    assert resp.status_code == 200
    data = resp.json()
    assert data["degraded_mode"] is False
    assert data["degraded_reason"] is None
    assert data["similar_patients_summary"]["k_neighbors"] == 20
    assert len(data["similar_patients_summary"]["top_neighbors"]) > 0
    assert 0.0 <= data["calibrated_risk_probability"] <= 1.0


