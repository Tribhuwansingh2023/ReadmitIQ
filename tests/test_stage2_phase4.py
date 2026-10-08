import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.drift import calculate_psi, classify_drift_severity
from src.explain import ModelExplainer, format_clinical_reason
from src.final_test_eval import LOCK_FILE_PATH, run_final_test_evaluation


def test_psi_identical_distributions():
    """Verify PSI is 0.0 for identical distributions."""
    rng = np.random.RandomState(42)
    x = rng.normal(0, 1, 1000)
    psi = calculate_psi(x, x)
    assert psi < 0.01, f"Expected near-zero PSI for identical distributions, got {psi}"
    assert classify_drift_severity(psi) == "Minimal (Stable)"


def test_psi_distribution_shift():
    """Verify PSI detects significant distribution shifts."""
    rng = np.random.RandomState(42)
    x1 = rng.normal(0, 1, 1000)
    x2 = rng.normal(2, 1, 1000)  # Significant mean shift
    psi = calculate_psi(x1, x2)
    assert psi > 0.25, f"Expected high PSI for shifted distributions, got {psi}"
    assert classify_drift_severity(psi) == "High (Alert)"


def test_clinical_reason_formatting():
    """Verify plain-language clinical explanation generation."""
    reason_inp = format_clinical_reason("number_inpatient", 3, +0.35)
    assert "inpatient" in reason_inp.lower()
    assert "raises risk" in reason_inp.lower()

    reason_zero_inp = format_clinical_reason("number_inpatient", 0, -0.20)
    assert "lowers risk" in reason_zero_inp.lower()

    reason_rag = format_clinical_reason("rag_struct_rate_k20", 0.35, +0.25)
    assert "similar historical patients" in reason_rag.lower()
    assert "raises risk" in reason_rag.lower()


def test_model_explainer_single_patient():
    """Verify ModelExplainer generates top risk increasing and reducing factors."""
    explainer = ModelExplainer()
    sample_patient = {
        "number_inpatient": 2,
        "number_emergency": 0,
        "number_outpatient": 1,
        "total_prior_visits": 3,
        "time_in_hospital": 4,
        "num_medications": 12,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "number_diagnoses": 7,
        "age_num": 65,
        "polypharmacy": 0,
        "n_active_drugs": 2,
        "n_dose_changes": 0,
        "labs_per_day": 10.0,
        "meds_per_day": 3.0,
        "a1c_tested": 0,
        "rag_struct_rate_k20": 0.20,
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
        "insulin": "No",
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
    result = explainer.explain_patient(sample_patient, top_k=3)
    assert "top_risk_increasing" in result
    assert "top_risk_reducing" in result
    assert len(result["top_risk_increasing"]) <= 3
    assert len(result["top_risk_reducing"]) <= 3


def test_eval_lock_enforcement():
    """Verify final test evaluation raises RuntimeError if lock file exists."""
    # Ensure lock exists or simulate lock
    was_locked = LOCK_FILE_PATH.exists()
    if not was_locked:
        LOCK_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE_PATH.write_text("TEST_LOCK_SIMULATION")

    try:
        with pytest.raises(RuntimeError, match="LOCKED"):
            run_final_test_evaluation(force_ignore_lock=False)
    finally:
        if not was_locked and LOCK_FILE_PATH.exists():
            LOCK_FILE_PATH.unlink()
