import sys
from pathlib import Path
from typing import Any, Dict, List, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from src.config import ALL_DRUG_COLS


def format_stay_band(days: int) -> str:
    """Describe hospital length of stay in categorical bands."""
    if days <= 2:
        return "short stay (1-2 days)"
    elif days <= 5:
        return "moderate stay (3-5 days)"
    elif days <= 9:
        return "extended stay (6-9 days)"
    return "prolonged stay (10 or more days)"


def format_inpatient_band(inp: int) -> str:
    """Describe prior inpatient admissions in clinical bands."""
    if inp == 0:
        return "no prior inpatient admissions in the prior year"
    elif inp == 1:
        return "1 prior inpatient admission in the prior year"
    elif inp == 2:
        return "2 prior inpatient admissions in the prior year"
    return "3 or more inpatient admissions in the prior year"


def format_emergency_band(er: int) -> str:
    """Describe emergency department visits in clinical bands."""
    if er == 0:
        return "no emergency visits"
    elif er == 1:
        return "1 prior emergency department encounter"
    return f"frequent emergency utilization ({er} visits)"


def format_a1c(a1c_val: Any) -> str:
    """Express HbA1c test result in clear clinical words."""
    val = str(a1c_val).strip()
    if val in ["None", "none", "", "nan"]:
        return "A1C not measured"
    elif val == "Norm":
        return "A1C in normal range"
    elif val == ">7":
        return "elevated A1C above 7 percent"
    elif val == ">8":
        return "markedly elevated A1C above 8 percent"
    return f"A1C level {val}"


def format_glucose(glu_val: Any) -> str:
    """Express serum glucose result in clear clinical words."""
    val = str(glu_val).strip()
    if val in ["None", "none", "", "nan"]:
        return "blood glucose serum not measured"
    elif val == "Norm":
        return "blood glucose serum normal"
    elif val == ">200":
        return "elevated blood glucose above 200 mg/dL"
    elif val == ">300":
        return "severely elevated blood glucose above 300 mg/dL"
    return f"blood glucose level {val}"


def format_meds_band(num_meds: int) -> str:
    """Express medication count in polypharmacy categories."""
    if num_meds < 10:
        return f"regimen of {num_meds} medications"
    elif num_meds < 20:
        return f"polypharmacy regimen of {num_meds} medications"
    return f"heavy polypharmacy burden of {num_meds} medications"


def format_labs_band(num_labs: int) -> str:
    """Express laboratory volume in testing bands."""
    if num_labs < 30:
        return f"routine lab testing ({num_labs} procedures)"
    elif num_labs < 60:
        return f"moderate lab monitoring ({num_labs} procedures)"
    return f"intensive lab monitoring ({num_labs} procedures)"


def format_active_drugs(row: Union[pd.Series, Dict[str, Any]]) -> str:
    """Extract active diabetes medications and dose adjustments."""
    active = []
    for drug in ALL_DRUG_COLS:
        val = str(row.get(drug, "No")).strip()
        if val in ["Steady", "Up", "Down"]:
            if val == "Steady":
                active.append(f"{drug} maintained steady")
            elif val == "Up":
                active.append(f"{drug} titrated upward")
            elif val == "Down":
                active.append(f"{drug} titrated downward")
    return ", ".join(active) if active else "no active diabetes medications"


def generate_single_clinical_note(row: Union[pd.Series, Dict[str, Any]], style_idx: int = 0) -> str:
    """Generate a single consistent clinical discharge note without label leakage.

    Uses ONE uniform grammatical template for all patients so semantic embeddings
    reflect strictly clinical and utilization characteristics, not template variation.
    Excludes race and all target/outcome words.
    """
    if isinstance(row, pd.Series):
        r = row.to_dict()
    else:
        r = row

    age = int(r.get("age_num", 55))
    gender = str(r.get("gender", "patient")).lower()
    time_hosp = int(r.get("time_in_hospital", 3))
    stay_str = format_stay_band(time_hosp)

    adm_src = str(r.get("admission_source_group", "referral")).replace("_", " ")
    adm_type = str(r.get("admission_type_group", "urgent")).replace("_", " ")
    d_disp = str(r.get("discharge_disposition_group", "home")).replace("_", " ")

    d1 = str(r.get("diag_1_group", "other"))
    d2 = str(r.get("diag_2_group", "other"))
    d3 = str(r.get("diag_3_group", "other"))

    inp_str = format_inpatient_band(int(r.get("number_inpatient", 0)))
    er_str = format_emergency_band(int(r.get("number_emergency", 0)))

    labs_str = format_labs_band(int(r.get("num_lab_procedures", 30)))
    meds_str = format_meds_band(int(r.get("num_medications", 10)))

    a1c_str = format_a1c(r.get("A1Cresult", "None"))
    glu_str = format_glucose(r.get("max_glu_serum", "None"))

    active_drugs_str = format_active_drugs(r)
    n_changes = int(r.get("n_dose_changes", 0))
    dose_change_str = (
        f"{n_changes} medication dose adjustments during stay"
        if n_changes > 0
        else "no dose modifications"
    )

    note = (
        f"Clinical discharge documentation. "
        f"Patient is an adult {gender} aged {age} hospitalized for a {stay_str}. "
        f"Admission source: {adm_src} under {adm_type} status. "
        f"Primary diagnostic group: {d1}, accompanied by secondary conditions {d2} and {d3}. "
        f"Prior healthcare utilization: {inp_str}, and {er_str}. "
        f"Laboratory diagnostic profile: {labs_str}. "
        f"Metabolic and glycemic status: {a1c_str}, and {glu_str}. "
        f"Inpatient pharmacotherapy: {meds_str}; diabetes medical therapies: {active_drugs_str}; {dose_change_str}. "
        f"Discharge disposition destination: {d_disp}."
    )
    return note


def generate_notes_for_dataframe(df: pd.DataFrame) -> List[str]:
    """Generate clinical notes rapidly for all rows in a DataFrame."""
    records = df.to_dict(orient="records")
    return [generate_single_clinical_note(r) for r in records]
