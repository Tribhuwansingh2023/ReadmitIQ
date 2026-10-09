# ReadmitIQ Presentation Slide Content

Each section below corresponds to one presentation slide for the hackathon pitch, containing exact empirical metrics extracted directly from saved project evaluation artifacts (`reports/metrics/`).

---

## Slide 1: The Problem — The Readmission Bottleneck
- **Title:** The $17 Billion Readmission Challenge in Diabetic Care
- **Core Problem:** 30-day hospital readmission is a critical indicator of post-acute care failure and healthcare cost. For diabetic patients, complex polypharmacy and chronic comorbidities create high vulnerability.
- **The Clinical Barrier:**
  1. *Resource Constraints:* Hospital care coordinators can only intensively follow up with 5% to 10% of discharged patients.
  2. *The Trust Deficit:* Attending physicians routinely ignore black-box statistical scores that lack explainable clinical precedent.
- **The ReadmitIQ Solution:** A calibrated, capacity-aware triage command center that augments gradient boosted risk scoring with **Structured Retrieval-Augmented Generation (RAG)** to provide case-based precedent reasoning.

---

## Slide 2: Data Foundations & Leakage Traps
- **Title:** Real-World Clinical Data: Rigorous Hygiene Without Leakage
- **Dataset:** UCI Diabetes 130-US Hospitals (1999–2008), 101,766 raw encounters across 71,518 unique patients.
- **Eligibility Filtering:** 99,340 eligible encounters after excluding in-hospital mortality and hospice discharges.
- **The Multi-Encounter Patient Leakage Trap:**
  - 1.42 encounters per patient on average (max 40 encounters).
  - Random row-based splitting leaks patient-specific baselines across train and test.
  - **Our Solution:** Grouped, patient-level splitting (`patient_nbr`) ensuring **0% patient overlap**:
    - **TRAIN:** 69,535 encounters (11.41% readmission rate).
    - **VAL:** 14,902 encounters (11.23% readmission rate).
    - **TEST:** 14,903 encounters (11.37% readmission rate).
- **Target Definition:** Binary $<30$-day readmission (`readmitted == '<30'`). Strict prevention of future encounter contamination.

---

## Slide 3: System Architecture & Structured RAG
- **Title:** Beyond Black Boxes: Case-Based Precedent Retrieval
- **Tabular Pipeline:** 41 engineered features (ICD-9 clinical agrupations, polypharmacy flags, labs-per-day, prior inpatient trajectories) expanded to 141 dense dimensions with rare-category collapsing (<150 columns).
- **Structured RAG Engine:**
  - 64-dimensional TruncatedSVD projection fit strictly on TRAIN features.
  - L2-normalized Inner Product FAISS index indexing 69,535 training patient vectors.
  - Queries retrieve the $k=20$ nearest historical clinical cases in under 2 milliseconds.
  - Out-of-Fold (OOF) 5-fold cross-validation on TRAIN prevents self-retrieval label leakage.
- **Clinical Rationale:** Doctors reason by precedent: *"What happened to the last 20 patients like this?"*

---

## Slide 4: Empirical Results & The Honest RAG Finding
- **Title:** Validation Benchmark & Empirical Hypothesis Testing
- **Model Ablation Table (Full 14,902 Validation Split):**
  - **Logistic Regression (Base):** AUC 0.6611 (CI: [0.6479, 0.6744]), Lift@10%: 2.14x
  - **Random Forest (Base):** AUC 0.6672 (CI: [0.6541, 0.6804]), Lift@10%: 2.18x
  - **LightGBM (Base):** AUC 0.6722 (CI: [0.6593, 0.6854]), Lift@10%: 2.22x
  - **LightGBM + Structured RAG (Frozen Winner):** AUC **0.6731** (CI: [0.6602, 0.6863]), Lift@10%: **2.22x**
- **The Honest Scientific Callout:**
  - Paired Bootstrap Delta AUC on TEST: **+0.0004** (95% CI: **[-0.0025, +0.0031]**).
  - *The confidence interval crosses zero.* RAG does not offer statistically significant predictive AUC gain over 41 tabular features alone.
  - **Takeaway:** We report this honestly rather than exaggerating. RAG's true value is clinical trust, not marginal ranking lift.

---

## Slide 5: Held-Out Test Evaluation & Calibration
- **Title:** Locked Test Evaluation: Rigorous Calibration & Operational Lift
- **Locked Test Evaluation (N = 14,903, Evaluated Once under `test_eval.lock`):**
  - **AUC-ROC:** **0.6590** (95% Bootstrap CI: **[0.6458, 0.6728]**).
  - **F1 Score:** **0.2721** at frozen optimal threshold (0.1495).
  - **Precision @ Top-10% Workload:** **23.76%** (Lift: **2.09x** over 11.37% random prevalence).
  - **Recall @ Top-10% Workload:** **20.9%** of all cohort readmissions captured.
- **Probability Calibration Impact:**
  - Raw uncalibrated ensemble probabilities produce a Brier score of ~0.22.
  - Isotonic regression fit on Train 5-fold OOF reduced Test Brier score to **0.0973** (55.6% error reduction).
  - Expected Calibration Error (ECE) is **0.0039** (<0.4% reliability deviation).

---

## Slide 6: Product Demo — Care-Team Command Center
- **Title:** ReadmitIQ: Actionable Clinical Decision Support
- **Screen 1: Cohort Command Center:**
  - Dynamic capacity slider (1% to 30%).
  - Shows exact care-coordinator staffing impact: At 10% capacity, 1,490 patients are flagged, capturing 354 true readmissions with 23.76% precision.
- **Screen 2: Patient Profile & Explainability:**
  - Circular calibrated risk gauge with color-coded operational risk tiers (Critical, High, Medium, Low).
  - Plain-language SHAP factor attribution (*"3 prior inpatient admissions in the past year raises risk"*).
  - Precedent table showing top-5 historical matches with their diagnoses and observed 30-day outcomes.
- **Screen 3 & 4: Model Lab & Fairness Dashboard:**
  - Real-time transparency for hospital governance and oversight.

---

## Slide 7: Algorithmic Fairness & Model Governance
- **Title:** Demographic Parity & Distribution Drift Telemetry
- **Subgroup Fairness Audit (Held-Out Test Split):**
  - Comprehensive audits across Race (Caucasian, AfricanAmerican, Other), Gender, and Age bands.
  - Disparate impact and True Positive Rate ratios remain stable across reliable cohorts (N >= 100).
- **Fairness by Blindness Comparison:**
  - Retrained parallel model with `race` and `gender` completely excluded.
  - Blinded Test AUC: **0.6583** vs Main Model **0.6590** (Delta: **-0.0007**).
  - *Conclusion:* Removing demographic attributes causes zero meaningful loss in predictive performance, proving the model is driven by clinical trajectories rather than demographic bias.
- **Population Stability Index (PSI):**
  - All monitored clinical features exhibit PSI < 0.05 (well below 0.10 stable threshold). Score distribution PSI = 0.0000.

---

## Slide 8: Real-World Limitations & Next Steps
- **Title:** Clinical Limitations & Deployment Roadmap
- **Limitations:**
  - *Historical Data:* 1999–2008 cohort. Modern SGLT2 inhibitors and GLP-1 agonists not reflected.
  - *Single Network Censoring:* Encounters only capture readmissions within participating hospitals.
  - *Lack of Continuous Labs & Vitals:* No outpatient blood pressure, BMI, or eGFR.
- **Roadmap & Production Next Steps:**
  - Direct EHR FHIR API integration for automated discharge summary ingestion.
  - Prospective silent-mode clinical pilot in diabetic step-down units.
  - Cost-benefit optimization modeling linking intervention cost against readmission penalty savings.
