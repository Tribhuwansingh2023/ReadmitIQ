import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from src.config import ALL_DRUG_COLS


def format_active_drugs(row: pd.Series) -> str:
    """Extract list of active diabetic medications and their dose directions."""
    active = []
    for drug in ALL_DRUG_COLS:
        if drug in row.index:
            val = str(row[drug]).strip()
            if val in ["Steady", "Up", "Down"]:
                if val == "Steady":
                    active.append(drug)
                else:
                    active.append(f"{drug} ({val.lower()})")
    return ", ".join(active) if active else "none"


def generate_single_clinical_note(row: pd.Series, style_idx: int = 0) -> str:
    """Generate a realistic synthetic clinical discharge note without label leakage.

    Uses rotating narrative templates to ensure natural sentence variation while
    strictly excluding any reference to readmission, hospital outcomes, or target labels.
    """
    age = int(row.get("age_num", 55))
    gender = str(row.get("gender", "Patient"))
    time_hosp = int(row.get("time_in_hospital", 3))
    adm_src = str(row.get("admission_source_group", "referral")).replace("_", " ")
    adm_type = str(row.get("admission_type_group", "emergency urgent")).replace("_", " ")
    d_disp = str(row.get("discharge_disposition_group", "home")).replace("_", " ")

    d1 = str(row.get("diag_1_group", "other"))
    d2 = str(row.get("diag_2_group", "other"))
    d3 = str(row.get("diag_3_group", "other"))

    prior_visits = int(row.get("total_prior_visits", 0))
    prior_inp = int(row.get("number_inpatient", 0))
    prior_er = int(row.get("number_emergency", 0))

    labs = int(row.get("num_lab_procedures", 30))
    meds = int(row.get("num_medications", 10))
    a1c = str(row.get("A1Cresult", "None"))
    glu = str(row.get("max_glu_serum", "None"))

    active_drugs_str = format_active_drugs(row)
    n_active = int(row.get("n_active_drugs", 0))
    n_changes = int(row.get("n_dose_changes", 0))

    if style_idx % 4 == 0:
        note = (
            f"Clinical Course & Discharge Summary: {age}-year-old {gender.lower()} admitted via {adm_src} "
            f"under {adm_type} status. Inpatient length of stay: {time_hosp} days. "
            f"Primary clinical assessment was {d1}, with secondary assessments of {d2} and {d3}. "
            f"In the prior 12 months, patient recorded {prior_visits} total visits ({prior_inp} inpatient, {prior_er} emergency). "
            f"Laboratory profile: {labs} procedures completed. Glycemic assessment: HbA1c result {a1c}, glucose serum {glu}. "
            f"Medical regimen: {meds} inpatient medications administered, including {n_active} diabetes agents ({active_drugs_str}) "
            f"with {n_changes} dosage adjustments. Patient discharged to {d_disp}."
        )
    elif style_idx % 4 == 1:
        note = (
            f"Discharge Handover: {gender}, age {age}. Hospital stay was {time_hosp} days following {adm_type} "
            f"admission through {adm_src}. Primary diagnosis: {d1}. Additional conditions addressed: {d2}, {d3}. "
            f"Prior healthcare utilization includes {prior_visits} prior contacts ({prior_inp} hospitalizations, {prior_er} ER encounters). "
            f"Workup involved {labs} diagnostic lab tests. Metabolic monitoring showed HbA1c {a1c} and serum glucose {glu}. "
            f"Inpatient pharmacotherapy comprised {meds} medications with {n_active} diabetic therapies ({active_drugs_str}) "
            f"and {n_changes} titration adjustments. Discharge disposition: {d_disp}."
        )
    elif style_idx % 4 == 2:
        note = (
            f"Physician Summary: Patient is a {age}-year-old {gender.lower()} with a {time_hosp}-day stay for {d1}. "
            f"Secondary diagnostic impressions include {d2} and {d3}. Admission pathway was {adm_src} ({adm_type}). "
            f"Prior year healthcare visits totaled {prior_visits} (inpatient admissions: {prior_inp}, emergency: {prior_er}). "
            f"Total clinical labs: {labs}. Glycemic control evaluation: HbA1c {a1c}, random glucose {glu}. "
            f"Pharmacologic management included {meds} medications with {n_active} active antidiabetic treatments ({active_drugs_str}) "
            f"and {n_changes} dosage modifications. Patient successfully transitioned to {d_disp}."
        )
    else:
        note = (
            f"Hospital Transition Documentation: Demographics: {gender.lower()}, {age} years old. "
            f"Admission mode: {adm_type} via {adm_src}. Total hospital stay: {time_hosp} days. "
            f"Diagnoses recorded: primary {d1}; secondary {d2}; tertiary {d3}. "
            f"Prior year encounters: {prior_visits} visits ({prior_inp} inpatient, {prior_er} emergency). "
            f"Diagnostic evaluation included {labs} laboratory procedures. Glycemic indices: HbA1c {a1c}, glucose {glu}. "
            f"Therapeutic plan: {meds} medications administered ({n_active} diabetic agents: {active_drugs_str}; {n_changes} dose changes). "
            f"Discharged status: {d_disp}."
        )

    return note


def generate_notes_for_dataframe(df: pd.DataFrame) -> List[str]:
    """Generate synthetic clinical notes for each row in a DataFrame."""
    notes = []
    for idx, (_, row) in enumerate(df.iterrows()):
        notes.append(generate_single_clinical_note(row, style_idx=idx))
    return notes
