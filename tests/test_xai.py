"""Comprehensive test suite for ReadmitIQ Explainable AI (XAI v2) layer.

Covers:
1. Exact TreeSHAP additivity (< 1e-6 tolerance) on frozen model.
2. Clinical grouping covers all 42 features exactly once.
3. Unrecorded / missing entries segregated to 'Information not recorded' and never in top clinical drivers.
4. Calibrated waterfall telescopes strictly to final calibrated risk.
5. Strength thresholds loaded from saved validation quantiles.
6. Contrastive tier explanations are deterministic and valid.
7. Constrained counterfactual adjustments only modify allowed features to training-observed values.
8. Retrieval evidence agreement rule on planted test cases.
9. Local reliability bins sum to validation sample with valid Wilson CIs.
10. Backward-compatible API POST /explain and GET /xai/global endpoints.
11. Verification that no test split data or labels are accessed by XAI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.service import get_service
from src.config import BASE_DIR, DATA_DIR, REPORTS_DIR
from src.xai.config import (
    FEATURE_GROUPS,
    FEATURE_MODIFIABILITY,
)
from src.xai.local import LocalExplainerV2


@pytest.fixture(scope="module")
def local_explainer():
    return LocalExplainerV2()


@pytest.fixture(scope="module")
def sample_validation_encounters():
    val_path = DATA_DIR / "processed" / "val.parquet"
    if val_path.exists():
        df = pd.read_parquet(val_path)
        return df.head(10).to_dict(orient="records")
    # Synthetic fallback
    return [
        {
            "number_inpatient": 2,
            "time_in_hospital": 4,
            "num_medications": 12,
            "num_lab_procedures": 35,
            "number_diagnoses": 7,
            "payer_code": "Missing",
            "A1Cresult": "None",
            "insulin": "Up",
            "change": "Ch",
            "diabetesMed": "Yes",
            "rag_struct_rate_k20": 0.35,
        }
    ]


def test_additivity_and_margin(local_explainer, sample_validation_encounters):
    """Assert additivity: base value + sum of contributions equals raw margin (< 1e-6)."""
    for enc in sample_validation_encounters:
        res = local_explainer.explain_patient(enc)
        assert res["additivity_verification"]["passed"] is True
        assert res["additivity_verification"]["violation_magnitude"] < 1e-6


def test_feature_grouping_covers_all_features_once(local_explainer):
    """Every feature in the frozen model must belong to exactly one clinical group."""
    all_cols = local_explainer.all_cols
    seen_features = set()

    for g_id, g_meta in FEATURE_GROUPS.items():
        assert "name" in g_meta
        assert "rationale" in g_meta
        assert len(g_meta["rationale"]) > 10
        for feat in g_meta["features"]:
            assert feat in all_cols, f"Group feature '{feat}' not in model features."
            assert feat not in seen_features, f"Feature '{feat}' duplicated across groups."
            seen_features.add(feat)

    assert seen_features == set(
        all_cols
    ), "Grouping does not cover all 42 model features exactly once."


def test_missing_and_unrecorded_never_in_clinical_drivers(local_explainer):
    """Features with unrecorded/missing values must never appear in top clinical drivers."""
    patient = {
        "number_inpatient": 3,
        "time_in_hospital": 5,
        "payer_code": "Missing",
        "medical_specialty": "Missing",
        "A1Cresult": "None",
        "max_glu_serum": "None",
        "rag_struct_rate_k20": 0.40,
    }
    res = local_explainer.explain_patient(patient)
    unrecorded = res["information_not_recorded"]
    unrecorded_names = {u["feature"] for u in unrecorded}

    assert "payer_code" in unrecorded_names
    assert "A1Cresult" in unrecorded_names
    for item in unrecorded:
        assert "neutral_text" in item
        assert "risk factor" not in item["neutral_text"].lower()

    # Verify absence from top drivers
    top_inc = {d["feature"] for d in res["top_risk_increasing"]}
    top_dec = {d["feature"] for d in res["top_risk_reducing"]}
    assert not (
        top_inc & unrecorded_names
    ), "Unrecorded features appeared in top risk increasing drivers."
    assert not (
        top_dec & unrecorded_names
    ), "Unrecorded features appeared in top risk reducing drivers."


def test_waterfall_telescopes_to_calibrated_risk(local_explainer, sample_validation_encounters):
    """Waterfall steps must telescope exactly to the final calibrated risk probability."""
    enc = sample_validation_encounters[0]
    res = local_explainer.explain_patient(enc)
    wf = res["calibrated_waterfall"]

    assert wf["telescoping_check_passed"] is True
    assert "order-dependent" in wf["order_dependent_label"].lower()
    steps = wf["steps"]
    assert len(steps) > 0

    accumulated = wf["reference_risk_at_base_margin"]
    for step in steps:
        step_effect = step["calibrated_prob_step_delta"]
        expected_cum = round(accumulated + step_effect, 4)
        assert abs(step["calibrated_prob_cumulative"] - expected_cum) < 1e-3
        accumulated = step["calibrated_prob_cumulative"]

    assert abs(wf["final_calibrated_prob"] - res["calibrated_prob"]) < 1e-4


def test_strength_thresholds_from_saved_file(local_explainer):
    """Strength labels must conform to saved thresholds.json quantiles."""
    th_path = REPORTS_DIR / "xai" / "thresholds.json"
    assert th_path.exists()
    with open(th_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    q_small = thresholds["small_threshold_max"]
    q_mod = thresholds["moderate_threshold_max"]

    enc = {
        "number_inpatient": 4,
        "time_in_hospital": 6,
        "rag_struct_rate_k20": 0.45,
    }
    res = local_explainer.explain_patient(enc)
    for driver in res["top_risk_increasing"] + res["top_risk_reducing"]:
        abs_s = driver["abs_shap_value"]
        if abs_s <= q_small:
            assert driver["strength"] == "small"
        elif abs_s <= q_mod:
            assert driver["strength"] == "moderate"
        else:
            assert driver["strength"] == "large"


def test_contrastive_deterministic(local_explainer):
    """Contrastive explanation for patient against lower tier reference must be deterministic."""
    enc = {
        "number_inpatient": 5,
        "time_in_hospital": 8,
        "num_medications": 22,
        "rag_struct_rate_k20": 0.50,
    }
    res1 = local_explainer.explain_patient(enc)
    res2 = local_explainer.explain_patient(enc)

    ct1 = res1["contrastive_explanation"]
    ct2 = res2["contrastive_explanation"]

    assert ct1["current_tier"] == ct2["current_tier"]
    assert ct1["compared_tier"] == ct2["compared_tier"]
    assert ct1["top_gap_features"] == ct2["top_gap_features"]


def test_counterfactual_constraints(local_explainer):
    """Counterfactual must only modify modifiable features and include disclaimer."""
    enc = {
        "number_inpatient": 1,
        "time_in_hospital": 4,
        "discharge_disposition_group": "SNF",
        "change": "No",
        "a1c_tested": 0,
        "rag_struct_rate_k20": 0.30,
    }
    res = local_explainer.explain_patient(enc)
    cf = res["counterfactual"]

    assert "disclaimer" in cf
    assert "Associative sensitivity simulation" in cf["disclaimer"]

    if cf.get("feasible_change_found"):
        tier_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        assert tier_order[cf["tier_after"]] < tier_order[cf["tier_before"]]
        assert len(cf["modifications"]) <= 2

        for mod in cf["modifications"]:
            feat = mod["feature"]
            mod_meta = FEATURE_MODIFIABILITY.get(feat, {})
            assert mod_meta.get("status") in ["modifiable", "partly_modifiable"]


def test_evidence_agreement_planted_cases(local_explainer):
    """Planted cases for evidence agreement consistency rules."""
    p_dict = {"number_inpatient": 1, "rag_struct_rate_k20": 0.25}

    # Case 1: Consistent (exact match within Wilson interval)
    res_cons = local_explainer._compute_evidence_agreement(
        calibrated_prob=0.25,
        patient_dict=p_dict,
        neighbor_summary={"neighbor_readmission_rate": 0.25, "k_neighbors": 20},
    )
    assert res_cons["agreement_status"] == "consistent"

    # Case 2: Inconsistent (large gap > 8 percentage points outside interval)
    res_incons = local_explainer._compute_evidence_agreement(
        calibrated_prob=0.45,
        patient_dict=p_dict,
        neighbor_summary={"neighbor_readmission_rate": 0.15, "k_neighbors": 20},
    )
    assert res_incons["agreement_status"] == "inconsistent"


def test_local_reliability_bins_sum_to_sample():
    """Reliability bins must sum to 5,000 validation patients with valid Wilson CIs."""
    rel_path = REPORTS_DIR / "xai" / "local_reliability.json"
    assert rel_path.exists()
    with open(rel_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["bins"]) == 10
    total_n = sum(b["n_patients"] for b in data["bins"])
    assert total_n == 5000

    for b in data["bins"]:
        low, high = b["wilson_ci_95"]
        assert 0.0 <= low <= high <= 1.0
        assert b["prob_range"][0] <= b["prob_range"][1]


def test_api_explain_backward_compatible():
    """POST /explain must preserve existing response schema, adding explanation_v2 when detail=full."""
    client = TestClient(app)
    payload = {
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 5,
        "admission_type_group": "Emergency",
        "discharge_disposition_group": "Home",
        "admission_source_group": "Emergency Room",
        "medical_specialty": "InternalMedicine",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Other",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
        "insulin": "No",
        "race": "Caucasian",
        "gender": "Female",
        "age_num": 65.0,
        "payer_code": "Missing",
        "rag_struct_rate_k20": 0.25,
    }

    # Standard call without detail
    resp_std = client.post("/explain?top_k=3", json=payload)
    assert resp_std.status_code == 200
    data_std = resp_std.json()
    assert "top_risk_increasing" in data_std
    assert "top_risk_reducing" in data_std
    assert "all_attributions" in data_std
    assert "base_value" in data_std
    assert "raw_margin" in data_std

    # Call with detail=full
    resp_full = client.post("/explain?top_k=3&detail=full", json=payload)
    assert resp_full.status_code == 200
    data_full = resp_full.json()
    assert "explanation_v2" in data_full
    v2 = data_full["explanation_v2"]
    assert v2 is not None
    assert "calibrated_waterfall" in v2
    assert "group_attributions" in v2
    assert "counterfactual" in v2
    assert "local_reliability" in v2


def test_api_xai_global_endpoint():
    """GET /xai/global and GET /v1/xai/global return complete artifacts or pending."""
    client = TestClient(app)

    # 1. Main endpoint
    res1 = client.get("/xai/global")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] in ["complete", "pending"]
    if data1["status"] == "complete":
        assert "global_importance" in data1
        assert "method_agreement" in data1
        assert "explanation_quality" in data1

    # 2. Router v1 endpoint
    res2 = client.get("/v1/xai/global")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == data1["status"]


def test_api_xai_global_pending_fallback():
    """When artifacts are missing, endpoint returns status pending gracefully."""
    svc = get_service()
    with patch.object(Path, "exists", return_value=False):
        res = svc.get_xai_global_summary()
        assert res["status"] == "pending"


def test_no_code_path_reads_test_split():
    """Verify strictly zero references to test.parquet in XAI analysis scripts."""
    xai_script = BASE_DIR / "scripts" / "run_xai_analysis.py"
    xai_local = BASE_DIR / "src" / "xai" / "local.py"

    with open(xai_script, "r", encoding="utf-8") as f:
        code_script = f.read()
    with open(xai_local, "r", encoding="utf-8") as f:
        code_local = f.read()

    assert "test.parquet" not in code_script
    assert "test.parquet" not in code_local
    assert "test_results.json" not in code_script
    assert "test_results.json" not in code_local
