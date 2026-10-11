# ReadmitIQ Data Card

## 1. Dataset Overview & Provenance
- **Dataset Name**: Diabetes 130-US Hospitals (1999–2008)
- **Curator & Lead Investigator**: Tribhuwan Singh ([@Tribhuwansingh2023](https://github.com/Tribhuwansingh2023), `webosingh93@gmail.com`)
- **Source**: UCI Machine Learning Repository / Center for Clinical and Translational Research, Virginia Commonwealth University
- **Citation**: Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International.
- **Task**: Binary classification predicting 30-day all-cause hospital readmission at the time of discharge for diabetic inpatients.
- **Ingestion & Verification**:
  - Raw Data File: `data/raw/diabetic_data.csv`
  - Ingestion Date: October 2026
  - Raw Encounter Count: 101,766 rows, 50 columns
  - **SHA-256 Verification Hash**:
    ```
    0689E7EC031237DC63031B938805C48377748761A3B26ACAB621567AFA24DF97
    ```

---

## 2. Cohort Definition & Exclusion Criteria

| Step | Criteria | Action | Rationale |
|---|---|---|---|
| 1 | Raw Encounters | 101,766 encounters, 71,518 unique patients | Full cohort of hospitalized patients diagnosed with diabetes. |
| 2 | Hospice / Mortality | Excluded discharge IDs `11, 13, 14, 19, 20, 21` (2,423 encounters) | Patients who expired or were discharged to hospice care are not eligible for 30-day community readmission. |
| 3 | Gender Unknown | Excluded 3 encounters where gender was `Unknown/Invalid` | Data hygiene; missing gender cannot be clinically imputed reliably. |
| **Final** | **Eligible Cohort** | **99,340 encounters across 69,987 unique patients** | High-integrity cohort for predictive modeling. |

---

## 3. Target Definition

The target variable is `readmitted_30d`:
$$\text{readmitted\_30d} = \begin{cases} 1 & \text{if } \text{readmitted} = \text{"<30"} \\ 0 & \text{if } \text{readmitted} \in \{\text{">30"}, \text{"NO"}\} \end{cases}$$

- **Class Prevalence**: ~11.39% positive across the entire cohort.
- **Clinical Relevance**: Inpatient readmissions within 30 days are penalizable under the CMS Hospital Readmissions Reduction Program (HRRP) and represent acute post-discharge care failures. Readmissions occurring after 30 days (`>30`) and non-readmissions (`NO`) are grouped into the negative class.

---

## 4. Patient-Level Split Strategy

To strictly prevent data leakage across multiple hospital visits of the same patient:
- **Splitting Method**: `StratifiedGroupKFold` grouped strictly by `patient_nbr`.
- **Partition Ratio**: 70% Train, 15% Validation, 15% Test.
- **Patient Isolation Guarantee**: Exactly 0 patient overlap across splits.
- **Stratification**: Target class prevalence is balanced across all splits.

| Split | Encounter Count | Patient Count | Positive Count | Positive Rate |
|---|---|---|---|---|
| **Train** | 69,535 (70.00%) | 48,988 | 7,919 | 11.388% |
| **Validation** | 14,902 (15.00%) | 10,520 | 1,697 | 11.388% |
| **Test** | 14,903 (15.00%) | 10,479 | 1,698 | 11.394% |

*Rule: The test split is sequestered and evaluated strictly once for final unbiased reporting.*

---

## 5. Feature Engineering & Preprocessing

### A. Clinical Mappings
- **Diagnosis Codes (ICD-9)**: `diag_1`, `diag_2`, and `diag_3` are mapped into 9 clinical categories:
  - `diabetes`: 250.xx
  - `circulatory`: 390–459, 785
  - `respiratory`: 460–519, 786
  - `digestive`: 520–579, 787
  - `genitourinary`: 580–629, 788
  - `neoplasms`: 140–239
  - `musculoskeletal`: 710–739
  - `injury`: 800–999
  - `other`: V/E codes and all remaining codes.
- **Age Interval**: Converted from ten-year intervals to numeric midpoints (e.g. `[0-10)` $\rightarrow$ 5, `[60-70)` $\rightarrow$ 65, ..., `[90-100)` $\rightarrow$ 95).
- **Administrative Groupings**:
  - `admission_type`: Emergency/Urgent (`IDs 1, 2, 7`), Elective (`ID 3`), Newborn (`ID 4`), and Other/Unknown (`IDs 5, 6, 8`).
    * *Notice on Other/Unknown*: IDs 5 (Trauma Center), 6 (NULL), and 8 (Not Mapped) encompass **10,396 rows (10.22% of the raw cohort)**. These were explicitly mapped to `other_unknown` to preserve sample size while avoiding arbitrary assignment.
  - `discharge_disposition`: Home (`ID 1`), Home Health (`IDs 6, 8`), Transferred Facility (`IDs 2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30`), Left AMA (`ID 7`), Other/Unknown (`Remaining`).
  - `admission_source`: Emergency Room (`ID 7`), Physician/Clinic Referral (`IDs 1, 2, 3`), Facility Transfer (`IDs 4, 5, 6, 10, 22, 25`), Other/Unknown (`Remaining`).

### B. Derived Clinical Metrics
- `total_prior_visits`: Sum of outpatient, emergency, and inpatient encounters in the preceding 12 months.
- `meds_per_day`: Number of medications divided by `time_in_hospital` (clipped at 1 day minimum).
- `labs_per_day`: Number of lab procedures divided by `time_in_hospital`.
- `polypharmacy`: Binary indicator for $\ge 10$ distinct concurrent medications during the encounter.
- `a1c_tested`: Binary indicator for whether an HbA1c test was ordered during the hospital stay.
- `n_active_drugs`: Count of active diabetic drugs prescribed (`Steady`, `Up`, `Down`) calculated prior to sparse drug filtering.
- `n_dose_changes`: Count of diabetic drugs undergoing dosage titration (`Up`, `Down`) during hospitalization.

### C. Dropped vs. Retained Drug Columns
Out of 23 original medication columns in the UCI schema:
- **Retained (11 Drugs)**: Formulate the core pharmaceutical features with active variation:
  `['metformin', 'repaglinide', 'nateglinide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin']`
- **Dropped (12 Drugs)**: Removed for lack of variance or extreme sparsity (>99.9% `"No"`):
  - *Zero-variance (2)*: `examide`, `citoglipton` (all records are `"No"`).
  - *Ultra-sparse (10)*: `chlorpropamide`, `acetohexamide`, `tolbutamide`, `miglitol`, `troglitazone`, `tolazamide`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`.
  - *Important*: These dropped drugs were factored into `n_active_drugs` and `n_dose_changes` before column removal so no clinical signal was lost.

### D. Leakage-Safe Imputation and Outlier Capping
- **Missing Categoricals**: `payer_code` (40% missing), `medical_specialty` (49% missing), and `race` (2.2% missing) imputed with explicit `"Missing"` level.
- **Lab Test Results**: `"None"` in `max_glu_serum` and `A1Cresult` represents "test not ordered", preserved as an informative categorical level rather than missing data.
- **Heavy-Tailed Outlier Capping**: Count features are clipped at their **99th percentile threshold calculated strictly on the TRAIN partition** (e.g. `number_inpatient`: 6, `total_prior_visits`: 11, `num_medications`: 43).
- **Identity Exclusion**: `encounter_id` and `patient_nbr` are strictly excluded from predictive model matrices.

---

## 6. Treatment of Sensitive Demographic Attributes (Race & Gender)
- **Role in Pipeline**: `race` (Caucasian, AfricanAmerican, Hispanic, Asian, Other, Missing) and `gender` (Female, Male) are retained in the base feature schema for benchmark auditing and comparative assessment.
- **Governance & Non-Discrimination**:
  - Race and gender are **not** used to steer clinical interventions.
  - In Phase 4, a formal demographic audit evaluates False Positive Rate Parity (FPR Ratio), True Positive Rate Parity (Equalized Odds), and Demographic Parity across racial subgroups.
  - In clinical deployment, care teams have the capability to run the prediction pipeline with demographic variables explicitly blinded/ablated.

---

## 7. Known Limitations
- **Single data source, historical period**: 130 US hospitals, 1999–2008. Treatment patterns, coding practices, and care pathways have changed since then (e.g., emergence of SGLT2 inhibitors and GLP-1 receptor agonists).
- **Readmissions are observed only within the same network**: A patient readmitted to an outside hospital system is labeled as not readmitted, so the true positive rate is likely higher than the recorded 11.4%.
- **Repeated encounters**: Patients average 1.42 encounters each (maximum 40). All eligible encounters are kept. Splitting is patient-level to prevent leakage, but encounters of the same patient within one split are correlated.
- **Missing context**: No continuous lab values (only A1C and glucose categories: Normal, >200, >300, >7, >8), no vital signs, no social determinants of health (SDOH), and no raw discharge-summary physician notes. The signal available for prediction is limited, so modest AUC (roughly 0.63–0.70) is expected.
- **Class imbalance**: About 11.4% positives. Handled with class weights and threshold tuning on the validation set.

---

## 8. Synthetic Clinical Notes Disclosure
Clinical notes are generated from the cleaned structured fields using templates with varied phrasing. They contain no label information and **add no new clinical information beyond the structured data**. They exist to enable the retrieval (RAG-style) step and the transformer baseline. Results from the notes should be read as a methodology demonstration, not as evidence of text-based clinical signal.

---

## 9. Evaluation Protocol
Thresholds and hyperparameters are chosen on the validation split only. The test split is evaluated once. Metrics: AUC-ROC (with bootstrap 95% CI), F1, Precision, Recall, and Precision@top-10% (capacity lift).

---

## 10. Complete Feature Dictionary (42 Features)

| # | Feature Name | Type | Source / Derivation | Clinical Description |
|---|---|---|---|---|
| 1 | `time_in_hospital` | Numeric | Raw encounter field | Length of stay in days (1–14). |
| 2 | `num_lab_procedures` | Numeric | Raw encounter field (capped at 84) | Number of laboratory tests performed during stay. |
| 3 | `num_procedures` | Numeric | Raw encounter field | Number of non-lab medical procedures performed. |
| 4 | `num_medications` | Numeric | Raw encounter field (capped at 43) | Number of distinct medications administered. |
| 5 | `number_outpatient` | Numeric | Raw encounter field (capped at 5) | Outpatient encounters in preceding 12 months. |
| 6 | `number_emergency` | Numeric | Raw encounter field (capped at 3) | Emergency visits in preceding 12 months. |
| 7 | `number_inpatient` | Numeric | Raw encounter field (capped at 6) | Inpatient admissions in preceding 12 months. |
| 8 | `number_diagnoses` | Numeric | Raw encounter field (capped at 9) | Number of secondary and primary diagnoses entered. |
| 9 | `age_num` | Numeric | Midpoint of `age` interval | Patient age estimate (years). |
| 10 | `total_prior_visits` | Numeric | Sum of prior visits (capped at 11) | Prior healthcare utilization density. |
| 11 | `meds_per_day` | Numeric | `num_medications / time_in_hospital` | Inpatient medication intensity per day. |
| 12 | `labs_per_day` | Numeric | `num_lab_procedures / time_in_hospital` | Diagnostic intensity per day. |
| 13 | `polypharmacy` | Numeric (Binary) | Indicator: `num_medications >= 10` | High medication load flag. |
| 14 | `a1c_tested` | Numeric (Binary) | Indicator: `A1Cresult != 'None'` | Adherence to diabetes glycemic monitoring guideline. |
| 15 | `n_active_drugs` | Numeric | Count of active diabetic drugs | Number of diabetic medications in active therapy. |
| 16 | `n_dose_changes` | Numeric | Count of drug dosage adjustments | Instability / titration in diabetic medication regimen. |
| 17 | `rag_readmit_rate_k20` | Numeric | Retrieval over 384d note embeddings | Mean 30-day readmission rate of $k=20$ nearest clinical neighbors. |
| 18 | `race` | Categorical | Demographic field | Self-reported racial background; audited for fairness. |
| 19 | `gender` | Categorical | Demographic field | Biological sex (`Female`, `Male`). |
| 20 | `admission_type_group` | Categorical | Grouped `admission_type_id` | `emergency_urgent`, `elective`, `newborn`, `other_unknown`. |
| 21 | `discharge_disposition_group` | Categorical | Grouped `discharge_disposition_id` | `home`, `home_health`, `transferred_facility`, `left_ama`, `other_unknown`. |
| 22 | `admission_source_group` | Categorical | Grouped `admission_source_id` | `emergency_room`, `referral`, `transfer`, `other_unknown`. |
| 23 | `diag_1_group` | Categorical | Grouped primary ICD-9 diagnosis | Primary diagnostic reason for admission (9 categories). |
| 24 | `diag_2_group` | Categorical | Grouped secondary ICD-9 diagnosis | First secondary comorbidity (9 categories). |
| 25 | `diag_3_group` | Categorical | Grouped tertiary ICD-9 diagnosis | Second secondary comorbidity (9 categories). |
| 26 | `max_glu_serum` | Categorical | Lab test result | Glucose serum indicator (`None`, `Norm`, `>200`, `>300`). |
| 27 | `A1Cresult` | Categorical | Lab test result | Glycated hemoglobin indicator (`None`, `Norm`, `>7`, `>8`). |
| 28 | `metformin` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 29 | `repaglinide` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 30 | `nateglinide` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 31 | `glimepiride` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 32 | `glipizide` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 33 | `glyburide` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 34 | `pioglitazone` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 35 | `rosiglitazone` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 36 | `acarbose` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 37 | `insulin` | Categorical | Active diabetic medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 38 | `glyburide-metformin` | Categorical | Active combination medication | Dosage status (`No`, `Steady`, `Up`, `Down`). |
| 39 | `change` | Categorical | Medication adjustment status | Indicator if diabetic medications changed (`Ch`, `No`). |
| 40 | `diabetesMed` | Categorical | Medication prescription status | Indicator if any diabetic medication prescribed (`Yes`, `No`). |
| 41 | `payer_code` | Categorical | Insurance payer category | Payer type code or `Missing` (e.g. `MC`, `MD`, `SP`, `BC`). |
| 42 | `medical_specialty` | Categorical | Admitting physician specialty | Specialty grouping or `Missing` (e.g. `InternalMedicine`, `Cardiology`). |
