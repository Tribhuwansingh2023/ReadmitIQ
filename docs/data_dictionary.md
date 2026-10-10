# ReadmitIQ Clinical Data Dictionary (42 Features)

ReadmitIQ evaluates encounters using 42 clinical features spanning demographic descriptors, encounter acuity, utilization history, diagnosis groups, laboratory tests, and medications.

---

## 1. Demographics & Utilization History
| Feature Name | Type | Range / Categories | Description |
| :--- | :--- | :--- | :--- |
| `age` | Categorical | `[0-10)` to `[90-100)` | Patient 10-year age bracket |
| `gender` | Categorical | `Female`, `Male`, `Unknown` | Administrative patient sex |
| `race` | Categorical | `Caucasian`, `AfricanAmerican`, `Hispanic`, `Asian`, `Other` | Self-reported race/ethnicity |
| `time_in_hospital` | Integer | $1 - 14$ days | Index admission length of stay |
| `number_inpatient` | Integer | $\ge 0$ | Number of prior inpatient admissions in preceding 12 months |
| `number_emergency` | Integer | $\ge 0$ | Number of emergency department visits in preceding 12 months |
| `number_outpatient` | Integer | $\ge 0$ | Number of outpatient clinic visits in preceding 12 months |
| `number_diagnoses` | Integer | $1 - 20$ | Total number of ICD diagnostic codes entered for index encounter |

---

## 2. Inpatient Procedures & Acuity
| Feature Name | Type | Range / Categories | Description |
| :--- | :--- | :--- | :--- |
| `num_lab_procedures` | Integer | $1 - 150$ | Total number of distinct lab tests ordered during stay |
| `num_procedures` | Integer | $0 - 10$ | Total non-lab surgical/diagnostic procedures performed |
| `num_medications` | Integer | $1 - 100$ | Total distinct pharmaceutical medications administered |
| `admission_type_group` | Categorical | `Emergency`, `Urgent`, `Elective`, `Other` | Initial hospital admission classification |
| `admission_source_group` | Categorical | `Emergency Room`, `Physician Referral`, `Transfer`, `Other` | Patient entry portal |
| `discharge_disposition_group` | Categorical | `Home`, `SNF`, `Home Health`, `Rehab`, `Other` | Post-acute discharge destination |
| `medical_specialty` | Categorical | `InternalMedicine`, `Cardiology`, `Surgery`, `Family`, `Other` | Admitting specialty service |
| `payer_code` | Categorical | `Medicare`, `Medicaid`, `Private`, `Self-Pay`, `Other` | Primary financial guarantor |

---

## 3. Clinical Diagnostic Groups (CCS Mappings)
| Feature Name | Type | Categories | Description |
| :--- | :--- | :--- | :--- |
| `diag_1_group` | Categorical | `Circulatory`, `Diabetes`, `Respiratory`, `Digestive`, `Genitourinary`, `Injury`, `Musculoskeletal`, `Neoplasms`, `Other` | Primary discharge diagnosis grouping |
| `diag_2_group` | Categorical | (Same categories as diag_1) | Secondary discharge diagnosis grouping |
| `diag_3_group` | Categorical | (Same categories as diag_1) | Tertiary discharge diagnosis grouping |

---

## 4. Glycemic Labs & Diabetes Medications
| Feature Name | Type | Categories | Description |
| :--- | :--- | :--- | :--- |
| `A1Cresult` | Categorical | `None`, `Norm`, `>7`, `>8` | Hemoglobin A1c glycemic control test result |
| `max_glu_serum` | Categorical | `None`, `Norm`, `>200`, `>300` | Peak serum blood glucose lab measurement |
| `diabetesMed` | Categorical | `Yes`, `No` | Any active diabetes medication prescribed |
| `change` | Categorical | `Ch`, `No` | Any titration or dosage change during admission |
| `insulin` | Categorical | `No`, `Steady`, `Up`, `Down` | Insulin dosage trajectory during admission |
| `metformin` | Categorical | `No`, `Steady`, `Up`, `Down` | Metformin prescription status |
| `glipizide` | Categorical | `No`, `Steady`, `Up`, `Down` | Sulfonylurea agent prescription |
| `glyburide` | Categorical | `No`, `Steady`, `Up`, `Down` | Sulfonylurea agent prescription |
| `glimepiride` | Categorical | `No`, `Steady`, `Up`, `Down` | Sulfonylurea agent prescription |
| `pioglitazone` | Categorical | `No`, `Steady`, `Up`, `Down` | Thiazolidinedione agent prescription |
| `rosiglitazone` | Categorical | `No`, `Steady`, `Up`, `Down` | Thiazolidinedione agent prescription |
| `acarbose` | Categorical | `No`, `Steady`, `Up`, `Down` | Alpha-glucosidase inhibitor prescription |
| `repaglinide` | Categorical | `No`, `Steady`, `Up`, `Down` | Meglitinide analog prescription |
| `nateglinide` | Categorical | `No`, `Steady`, `Up`, `Down` | Meglitinide analog prescription |
| `glyburide-metformin` | Categorical | `No`, `Steady`, `Up`, `Down` | Combination oral antidiabetic prescription |
