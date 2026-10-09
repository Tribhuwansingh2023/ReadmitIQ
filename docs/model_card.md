# ReadmitIQ Model Card: 30-Day Hospital Readmission Risk Predictor

## 1. Model Details
- **Model Name:** ReadmitIQ Frozen Production Classifier (`LightGBM + Structured RAG`)
- **Version:** 1.0.0 (Frozen Selection: `models/FROZEN_SELECTION.json`)
- **Model Architecture:** Gradient Boosted Decision Tree (LightGBM) with native categorical handling, augmented with a 64-dimensional Structured FAISS Inner Product retriever ($k=20$).
- **Calibration Method:** Isotonic Regression fitted strictly on 5-fold Out-Of-Fold (OOF) Train probability predictions.
- **License:** MIT License (Research & Clinical Decision Support Prototype).

---

## 2. Intended Use & Clinical Scope
- **Primary Clinical Purpose:** Risk stratification of diabetic inpatients at the time of discharge to prioritize care-management resources, post-discharge pharmacist consultations, and home-health outreach.
- **Target Population:** Adult diabetic patients admitted to acute-care facilities.
- **Out-of-Scope Uses:**
  - Autonomous diagnostic or treatment decisions without clinical provider review.
  - Triage in emergency department intake (model relies on in-hospital stay metrics).
  - Non-diabetic inpatient populations.

---

## 3. Training & Evaluation Data
- **Data Source:** UCI Diabetes 130-US Hospitals (1999–2008), 101,766 encounters across 71,518 unique patients.
- **Eligible Cohort:** 99,340 eligible encounters (excluding in-hospital deaths and hospice transfers).
- **Patient-Disjoint Splitting:**
  - **TRAIN Split:** 69,535 encounters (11.41% positive, 7,935 readmissions).
  - **VAL Split:** 14,902 encounters (11.23% positive, 1,673 readmissions).
  - **TEST Split:** 14,903 encounters (11.37% positive, 1,694 readmissions).
- **Data Leakage Safeguards:**
  - Grouped splitting by `patient_nbr` ensures 0% patient overlap between splits.
  - All scalers, encoders, and rare-category rules fit strictly on TRAIN data.
  - FAISS index contains only TRAIN encounters (69,535 rows).
  - Train RAG feature generated via 5-fold patient-disjoint OOF cross-validation.

---

## 4. Quantitative Performance Metrics

All metrics on the held-out TEST split were evaluated exactly once under `reports/metrics/test_eval.lock`:

| Metric | Held-Out Test Set (N = 14,903) | 95% Bootstrap CI | Notes |
|:---|:---:|:---:|:---|
| **AUC-ROC** | **0.6590** | [0.6458, 0.6728] | Bootstrap $B=1,000$ |
| **F1 Score** | **0.2721** | — | Evaluated at frozen threshold (0.1495) |
| **Precision @ Top 10%** | **23.76%** | [21.5%, 26.0%] | 354 readmitted out of 1,490 flagged |
| **Lift @ Top 10%** | **2.09x** | — | Baseline prevalence = 11.37% |
| **Recall @ Top 10%** | **20.9%** | — | 354 / 1,694 total readmissions captured |
| **Brier Score (Calibrated)** | **0.0973** | — | Raw uncalibrated Brier ~0.219 |
| **Expected Calibration Error (ECE)** | **0.0039** | — | <0.4% divergence across probability bins |

### Honest Empirical Finding: Structured RAG Impact
- **Test Paired Delta AUC (with RAG vs without RAG):** **+0.0004** (95% CI: **[-0.0025, +0.0031]**).
- **Hypothesis Test Result:** The paired bootstrap confidence interval crosses zero.
- **Finding:** Retrieving similar historical patients does not produce a statistically significant increase in predictive discriminative ranking over 41 tabular features alone.
- **Clinical Justification for Retention:** The FAISS retriever is retained because it provides crucial **case-based explainability** for clinicians, allowing care teams to inspect similar precedent cases and neighbor outcomes directly.

---

## 5. Algorithmic Fairness & Subgroup Analysis

Fairness audits were conducted across protected demographic attributes on the held-out TEST split:

| Demographic Subgroup | N | Prevalence | Subgroup AUC (95% CI) | Selection Rate (Top-10%) | TPR | Disparate Impact Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Caucasian** (Reference) | 11,102 | 11.4% | 0.6631 [0.648, 0.678] | 10.2% | 21.6% | 1.00x |
| **African American** | 2,827 | 11.8% | 0.6450 [0.617, 0.674] | 9.4% | 18.9% | 0.92x |
| **Hispanic** | 303 | 10.2% | 0.6410 [0.552, 0.728] | 8.3% | 19.4% | 0.81x |
| **Asian** | 98 | 8.2% | 0.7410 [0.584, 0.887] | 5.1% | 12.5% | Small Sample (<100) |
| **Female** (Reference) | 8,024 | 11.4% | 0.6620 [0.644, 0.680] | 10.1% | 21.3% | 1.00x |
| **Male** | 6,879 | 11.3% | 0.6554 [0.636, 0.675] | 9.9% | 20.4% | 0.98x |

### Demographic Blinded Model Comparison
- **Main Model AUC (with race/gender):** 0.6590
- **Blinded Model AUC (without race/gender):** 0.6583
- **Delta AUC:** -0.0007
- **Conclusion:** Omitting demographic variables causes negligible change in predictive performance (-0.07% AUC), indicating that predictions are driven by clinical trajectory features (prior hospitalizations, stay length, medication changes) rather than demographic proxying.

---

## 6. Interpretability & Explainability
- **Feature Attribution:** Global and local feature attributions are computed via TreeExplainer SHAP.
- **Plain-Language Translations:** Attributions are mapped to clinical sentences (e.g., *"3 prior inpatient admissions in the past year raises risk"*).
- **Precedent Transparency:** FAISS retrieval surfaces the top-5 historical encounters, their length of stay, diagnoses, and observed 30-day readmission status.

---

## 7. Limitations & Ethical Considerations
- **Historical Context:** Data is from 1999–2008. Treatment protocols, coding standards, and newer medications (GLP-1s, SGLT2i) are not represented.
- **Network Censoring:** Only readmissions to the same hospital system are captured.
- **Absence of Real-Time Vitals:** Clinical vitals (blood pressure, oxygenation) and daily laboratory values are absent from the dataset.
