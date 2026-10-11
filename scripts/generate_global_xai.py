import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from src.config import (
    METRICS_DIR,
    VAL_RAG_PARQUET,
    MODELS_DIR,
)
from src.train import prepare_lgbm_data

FEATURE_HUMAN_NAMES = {
    "number_inpatient": "Prior Inpatient Admissions",
    "discharge_disposition_group": "Discharge Destination",
    "medical_specialty": "Admitting Specialty",
    "total_prior_visits": "Total Prior Healthcare Visits",
    "diag_1_group": "Primary Diagnosis Category",
    "payer_code": "Primary Payer Category",
    "rag_struct_rate_k20": "Peer Readmission Rate (RAG Precedents)",
    "number_diagnoses": "Coded Diagnoses Count",
    "diag_2_group": "Secondary Diagnosis Category",
    "insulin": "Insulin Regimen",
    "diag_3_group": "Tertiary Diagnosis Category",
    "admission_source_group": "Admission Source",
    "time_in_hospital": "Length of Stay (Days)",
    "age_num": "Patient Age Midpoint",
    "num_medications": "Prescribed Medications Count",
    "num_lab_procedures": "Diagnostic Lab Procedures",
    "admission_type_group": "Admission Urgency Type",
    "n_active_drugs": "Active Diabetes Medications",
    "change": "Diabetes Medication Modified",
    "num_procedures": "Non-Lab Clinical Procedures",
    "labs_per_day": "Daily Lab Procedure Velocity",
    "meds_per_day": "Daily Medication Prescribing Velocity",
    "metformin": "Metformin Therapy",
    "number_emergency": "Prior Emergency Room Visits",
    "number_outpatient": "Prior Outpatient Clinic Visits",
    "race": "Patient Race / Ethnicity",
    "gender": "Patient Biological Sex",
    "diabetesMed": "Prescribed Diabetes Medication",
    "A1Cresult": "HbA1c Lab Test Result",
    "glipizide": "Glipizide Regimen",
    "glyburide": "Glyburide Regimen",
    "pioglitazone": "Pioglitazone Regimen",
    "glimepiride": "Glimepiride Regimen",
    "rosiglitazone": "Rosiglitazone Regimen",
    "polypharmacy": "Polypharmacy Flag (>16 Meds)",
    "max_glu_serum": "Serum Glucose Lab Result",
    "a1c_tested": "HbA1c Lab Drawn During Stay",
    "n_dose_changes": "Medication Dose Adjustments",
    "repaglinide": "Repaglinide Regimen",
    "nateglinide": "Nateglinide Regimen",
    "acarbose": "Acarbose Regimen",
    "glyburide-metformin": "Glyburide-Metformin Combination",
}

FEATURE_CLINICAL_DESCRIPTIONS = {
    "number_inpatient": "Number of inpatient hospitalizations in the 12 months preceding the index discharge.",
    "discharge_disposition_group": "Destination patient is discharged to (e.g. Home, Skilled Nursing, Home Health, AMA).",
    "medical_specialty": "Primary specialty of admitting or attending physician.",
    "total_prior_visits": "Cumulative count of previous emergency, inpatient, and outpatient encounters in past year.",
    "diag_1_group": "Primary ICD diagnosis category documented at admission.",
    "payer_code": "Primary expected healthcare payer or insurance plan (e.g. Medicare, Medicaid, Private).",
    "rag_struct_rate_k20": "Empirical 30-day readmission rate among top-20 structurally similar historical diabetes discharges.",
    "number_diagnoses": "Total distinct ICD diagnoses recorded on the encounter claim.",
    "diag_2_group": "Secondary comorbidity ICD diagnosis category.",
    "insulin": "Status of insulin management (No, Steady, Up, Down).",
    "diag_3_group": "Tertiary comorbidity ICD diagnosis category.",
    "admission_source_group": "Point of origin prior to inpatient admission (e.g. Emergency Room, Referral).",
    "time_in_hospital": "Total duration of inpatient stay in days.",
    "age_num": "Estimated midpoint age in years for patient demographic cohort.",
    "num_medications": "Count of distinct pharmaceutical agents administered during hospitalization.",
    "num_lab_procedures": "Total count of diagnostic lab tests performed during hospitalization.",
    "admission_type_group": "Clinical urgency of admission (e.g. Emergency, Urgent, Elective).",
    "n_active_drugs": "Number of active diabetic pharmaceutical agents prescribed.",
    "change": "Whether any diabetic medication was altered during the inpatient stay.",
    "num_procedures": "Total non-lab diagnostic or surgical procedures performed during stay.",
    "labs_per_day": "Ratio of lab tests performed per inpatient hospital day.",
    "meds_per_day": "Ratio of distinct medications prescribed per inpatient hospital day.",
    "metformin": "Status of metformin therapy (No, Steady, Up, Down).",
    "number_emergency": "Emergency department visits in the 12 months preceding the index discharge.",
    "number_outpatient": "Outpatient clinic visits in the 12 months preceding the index discharge.",
    "race": "Recorded racial or ethnic identity in clinical registry.",
    "gender": "Recorded biological sex in patient registration.",
    "diabetesMed": "Indicator whether any anti-diabetic medication was prescribed.",
    "A1Cresult": "Recorded glycated hemoglobin test result range during admission.",
    "glipizide": "Sulfonylurea medication regimen (No, Steady, Up, Down).",
    "glyburide": "Sulfonylurea medication regimen (No, Steady, Up, Down).",
    "pioglitazone": "Thiazolidinedione medication regimen (No, Steady, Up, Down).",
    "glimepiride": "Sulfonylurea medication regimen (No, Steady, Up, Down).",
    "rosiglitazone": "Thiazolidinedione medication regimen (No, Steady, Up, Down).",
    "polypharmacy": "Indicator whether patient received more than 16 distinct medications.",
    "max_glu_serum": "Peak serum glucose lab result category recorded during stay.",
    "a1c_tested": "Binary indicator confirming whether an HbA1c test was performed during the encounter.",
    "n_dose_changes": "Total count of diabetes medication titration events (Up or Down).",
    "repaglinide": "Meglitinide medication regimen (No, Steady, Up, Down).",
    "nateglinide": "Meglitinide medication regimen (No, Steady, Up, Down).",
    "acarbose": "Alpha-glucosidase inhibitor regimen (No, Steady, Up, Down).",
    "glyburide-metformin": "Combination oral agent therapy regimen (No, Steady, Up, Down).",
}


def build_global_xai_summary():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "global_xai_summary.json"

    print("Loading models and validation reference cohort...")
    model = joblib.load(MODELS_DIR / "final" / "model.joblib")
    with open(MODELS_DIR / "final" / "feature_list.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(MODELS_DIR / "final" / "serving_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    with open(METRICS_DIR / "test_results.json", "r", encoding="utf-8") as f:
        test_res = json.load(f)

    val_df = pd.read_parquet(VAL_RAG_PARQUET)
    num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]
    all_cols = meta["all_features"]

    X_val = prepare_lgbm_data(val_df, num_cols, cat_cols)
    print(f"Computing TreeSHAP pred_contrib for {len(val_df)} validation rows...")
    contribs = model.booster_.predict(X_val, pred_contrib=True)
    shap_vals = contribs[:, :-1]
    base_val = float(round(float(contribs[0, -1]), 6))

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    pos_shares = np.mean(shap_vals > 1e-4, axis=0)
    neg_shares = np.mean(shap_vals < -1e-4, axis=0)

    # Rank features
    sorted_idx = np.argsort(mean_abs)[::-1]

    feature_importances = []
    for rank, idx in enumerate(sorted_idx, start=1):
        col = all_cols[idx]
        is_num = col in num_cols
        series = val_df[col] if col in val_df.columns else pd.Series([])

        stats = {}
        if is_num and not series.empty:
            s_clean = pd.to_numeric(series, errors="coerce").dropna()
            stats = {
                "min": float(round(s_clean.min(), 2)),
                "p25": float(round(s_clean.quantile(0.25), 2)),
                "median": float(round(s_clean.median(), 2)),
                "p75": float(round(s_clean.quantile(0.75), 2)),
                "max": float(round(s_clean.max(), 2)),
                "mean": float(round(s_clean.mean(), 2)),
            }
        elif not series.empty:
            vc = series.value_counts(normalize=True).head(5).to_dict()
            stats = {
                "top_categories": {str(k): round(float(v), 3) for k, v in vc.items()}
            }

        feature_importances.append(
            {
                "rank": rank,
                "feature": col,
                "feature_name": FEATURE_HUMAN_NAMES.get(col, col),
                "description": FEATURE_CLINICAL_DESCRIPTIONS.get(col, f"Clinical feature {col}"),
                "mean_abs_shap": float(round(mean_abs[idx], 4)),
                "direction_distribution": {
                    "positive_risk_pct": float(round(pos_shares[idx] * 100, 1)),
                    "protective_risk_pct": float(round(neg_shares[idx] * 100, 1)),
                },
                "feature_type": "numeric" if is_num else "categorical",
                "summary_stats": stats,
            }
        )

    summary = {
        "model_name": test_res.get("model_name", "LightGBM + Structured RAG"),
        "git_commit_hash": test_res.get("git_commit_hash", "efa44003ae6ea2bc4243532def61ef50cf13630b"),
        "explainer_method": "Native LightGBM TreeSHAP (Additive Decision Tree Path Contribution)",
        "reference_cohort": {
            "name": "Validation Cohort (val_rag.parquet)",
            "sample_size": len(val_df),
            "positive_prevalence": float(round(float(val_df["readmitted_30d"].mean()), 4)),
            "time_split_window": "Temporal held-out partition of diabetic inpatient encounters (discharged 30d follow-up)",
            "leakage_protection": "Disjoint patient-level splitting; zero future data leakage.",
        },
        "output_scale": {
            "scale_type": "log_odds_margin",
            "base_value": base_val,
            "units": "Log-Odds Margin Space",
            "mathematical_definition": "phi_0 + sum(phi_i) == raw_margin; sigmoid(raw_margin) == raw_uncalibrated_probability",
            "clinical_scale_explanation": (
                "TreeSHAP attributions represent additive shifts in model log-odds margin relative to explainer baseline (-0.1960); "
                "they do NOT represent percentage-point changes in 30-day readmission risk."
            ),
        },
        "feature_importances": feature_importances,
        "calibration_summary": {
            "method": "Isotonic Regression (Non-parametric piecewise monotonic mapping)",
            "artifact": "models/final/calibrator.joblib",
            "validation_sample_size": thresholds["validation_distribution"]["sample_size"],
            "test_sample_size": test_res["test_cohort_size"],
            "ece": test_res["ece"],
            "brier_score": test_res["brier_score"],
            "test_auc_roc": test_res["auc_roc"],
            "test_auc_ci_95": test_res["auc_ci_95"],
            "optimal_f1_threshold": test_res["frozen_decision_threshold"],
        },
        "capacity_thresholds": thresholds.get("tier_definitions", {}),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Global XAI summary successfully generated at {out_path} ({len(feature_importances)} features ranked).")


if __name__ == "__main__":
    build_global_xai_summary()
