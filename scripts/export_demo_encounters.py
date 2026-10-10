"""Export sample non-PHI encounter CSV batch for testing CSV ingestion in the ReadmitIQ Web Desk.

Usage:
    python scripts/export_demo_encounters.py [output_path]
"""

import csv
import sys
from pathlib import Path

DEFAULT_OUT = Path("data/sample_intake_batch.csv")

SAMPLE_RECORDS = [
    {
        "external_ref": "ENC-DEMO-001",
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
    },
    {
        "external_ref": "ENC-DEMO-002",
        "age": "[50-60)",
        "gender": "Male",
        "race": "AfricanAmerican",
        "time_in_hospital": 3,
        "num_lab_procedures": 38,
        "num_procedures": 0,
        "num_medications": 14,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 1,
        "number_diagnoses": 6,
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency Room",
        "discharge_disposition_group": "Home",
        "medical_specialty": "Emergency",
        "payer_code": "Medicaid",
        "diag_1_group": "Diabetes",
        "diag_2_group": "Circulatory",
        "diag_3_group": "Other",
        "max_glu_serum": "None",
        "A1Cresult": "Norm",
        "diabetesMed": "Yes",
        "change": "No",
        "insulin": "No",
        "metformin": "Steady",
    },
    {
        "external_ref": "ENC-DEMO-003",
        "age": "[60-70)",
        "gender": "Male",
        "race": "Caucasian",
        "time_in_hospital": 2,
        "num_lab_procedures": 25,
        "num_procedures": 1,
        "num_medications": 7,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 4,
        "admission_type_group": "Elective",
        "admission_source_group": "Physician Referral",
        "discharge_disposition_group": "Home",
        "medical_specialty": "Cardiology",
        "payer_code": "Private",
        "diag_1_group": "Circulatory",
        "diag_2_group": "Other",
        "diag_3_group": "Other",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "diabetesMed": "No",
        "change": "No",
        "insulin": "No",
        "metformin": "No",
    },
]


def main():
    out_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SAMPLE_RECORDS[0].keys()))
        writer.writeheader()
        writer.writerows(SAMPLE_RECORDS)
    print(f"[SUCCESS] Exported {len(SAMPLE_RECORDS)} demo encounter records to: {out_file}")


if __name__ == "__main__":
    main()
