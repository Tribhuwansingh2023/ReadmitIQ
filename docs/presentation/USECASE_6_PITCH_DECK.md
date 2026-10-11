# SOA Use Case #6: Hospital Readmission Risk Prediction
## Project Title: ReadmitIQ — Supervised Learning & Clinical RAG Feature Engineering

- **Use Case:** #6 — Hospital Readmission Risk Prediction
- **Track:** Supervised Learning + RAG Feature Engineering — Binary Classification
- **Mentor:** Sulagna Tripathy
- **Live Backend:** https://readmitiq-wlw1.onrender.com (FastAPI on Render)
- **Live Frontend:** https://readmit-iq.vercel.app (React SPA on Vercel)
- **Interactive OpenAPI Docs:** https://readmitiq-wlw1.onrender.com/docs

---

## 10-Slide Presentation Deck

---

### Slide 1: Title & Use Case Overview
- **Category:** `SOA USE CASE #6 | HOSPITAL READMISSION RISK PREDICTION`
- **Slide Title:** ReadmitIQ: Supervised Readmission Prediction via Case-Based RAG Retrieval
- **Subtitle:** An End-to-End Clinical Decision Support Engine with Calibrated Triage & Evidence Retrieval
- **Key Points:**
  - Evaluated on UCI Diabetes 130-US Hospitals (N = 99,340 eligible encounters across 70,442 patients).
  - Primary Model: LightGBM + Structured RAG | Test AUC: 0.6590 [95% CI: 0.6458, 0.6728].
  - Operational Lift: 2.09x Lift at Top-10% Outreach Capacity (captures 20.9% of readmissions with 23.76% precision).
  - Isotonic Calibration: Expected Calibration Error (ECE) is 0.39% (Brier score reduced to 0.0973).
  - Scientific Integrity Pledge: Paired Bootstrap RAG Delta on test is +0.0004 (CI crosses zero); RAG provides clinical trust, not score inflation.

---

### Slide 2: Clinical Problem Statement & Operational Economics
- **Category:** `01. PROBLEM CONTEXT & HEALTHCARE ECONOMICS`
- **Slide Title:** The $26B Readmission Bottleneck: Why Manual Triage Fails
- **Key Points:**
  - **The HRRP Penalty Threat:** CMS penalties up to 3% of total inpatient Medicare revenue for excess 30-day readmissions.
  - **Fixed Care-Team Capacity:** Hospital transition nurse teams can only follow up with 5% to 10% of discharged patients.
  - **The Clinician Trust Gap:** Attending physicians routinely ignore black-box scores like "74% Risk" without peer precedents.
  - **The ReadmitIQ Solution:** Capacity-aware triage ranking with analogous past patient outcomes retrieved at discharge.

---

### Slide 3: Data Source & 4 Clinical Leakage Traps
- **Category:** `02. DATASET RIGOR & PREPROCESSING`
- **Slide Title:** Real-World Clinical Data: Defeating 4 Critical Leakage Traps
- **Key Points:**
  - **Data Source:** UCI Diabetes 130-US Hospitals (1999–2008), 101,766 raw encounters -> 99,340 eligible discharges.
  - **Trap 1: Multi-Encounter Patient Leak:** GroupKFold on `patient_nbr` ensuring 0% patient overlap across Train (69,535), Val (14,902), and Test (14,903).
  - **Trap 2: Mortality & Hospice Contamination:** Filtered 2,426 expired/hospice records (discharge codes 11, 13, 14, 19, 20, 21).
  - **Trap 3: Informative Missingness:** Preserved "None" as an explicit state for unmeasured HbA1c/glucose labs.
  - **Trap 4: ICD Sparsity:** Collapsed 700+ raw ICD-9 codes into 9 clinical organ systems and 141 dense features.

---

### Slide 4: Synthetic Clinical-Note Summaries & RAG Architecture
- **Category:** `03. RAG FEATURE ENGINEERING`
- **Slide Title:** The RAG Retrieval Step: Case Summaries to Historical Readmission Rate
- **Key Points:**
  - **Synthetic Note Generator:** Audited clinical narratives with zero target label tokens.
  - **Latent Embedding Space:** 64-dimensional TruncatedSVD projection fit strictly on Train.
  - **Historical Case Retrieval:** FAISS IndexFlatIP indexing 69,535 training encounters (<2ms search).
  - **OOF Target Masking:** 5-fold cross-validation on Train prevents self-retrieval label leakage.
  - **Engineered Feature (`rag_rate_k20`):** Empirical 30-day readmission rate among top-20 historical peers.

---

### Slide 5: Comprehensive Ablation Table (With vs. Without RAG)
- **Category:** `04. BENCHMARKING & METRIC ABLATION`
- **Slide Title:** Model Benchmark Across AUC-ROC, F1, Precision & Recall

| Model Architecture | RAG Feature | AUC-ROC [95% CI] | F1 Score | Precision | Recall | Top-10% Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Without RAG (Base)** | 0.6619 [0.6484, 0.6757] | 0.2770 | 19.14% | 50.15% | 2.09x |
| **Logistic Regression** | **With RAG Feature** | **0.6632** [0.6501, 0.6772] | **0.2766** | **18.66%** | **53.45%** | **2.13x** |
| **Random Forest** | **Without RAG (Base)** | 0.6671 [0.6547, 0.6804] | 0.2795 | 19.04% | 52.50% | 2.11x |
| **Random Forest** | **With RAG Feature** | **0.6674** [0.6548, 0.6806] | **0.2778** | **18.77%** | **53.45%** | **2.20x** |
| **LightGBM (GBDT)** | **Without RAG (Base)** | 0.6722 [0.6593, 0.6854] | 0.2819 | 19.70% | 49.50% | 2.25x |
| **LightGBM + RAG** *(Winner)* | **With RAG Feature** | **0.6731** [0.6602, 0.6863] | **0.2822** | **18.66%** | **57.87%** | **2.22x** |
| **Transformer (DistilBERT)** | **Clinical Note Embed** | *Validation Benchmark* | *0.2640* | *17.50%* | *51.20%* | *1.98x* |

---

### Slide 6: The Honest Scientific Finding (Bootstrap Hypothesis Testing)
- **Category:** `05. HYPOTHESIS TESTING & STATISTICAL RIGOR`
- **Slide Title:** Statistical Hypothesis Testing: What Does RAG Really Deliver?
- **Key Points:**
  - **1,000-Sample Paired Bootstrap Test:** Observed Delta AUC = +0.0004 [95% CI: -0.0025, +0.0031].
  - **Statistical Verdict:** The confidence interval crosses zero (p > 0.05).
  - **Scientific Honesty:** RAG does not offer statistically significant discrimination gain over 41 tabular features alone.
  - **True Healthcare Value:** Case precedent reasoning and physician trust, not metric inflation.

---

### Slide 7: Locked Test Evaluation & Capacity-Aware Triage
- **Category:** `06. INDEPENDENT TEST EVALUATION`
- **Slide Title:** Held-Out Test Results: Calibration & Operational Lift
- **Key Points:**
  - **Locked Test Split (N = 14,903):** AUC-ROC 0.6590 [95% CI: 0.6458, 0.6728] under `test_eval.lock`.
  - **Top-5% Capacity:** Precision 26.98% | 2.37x Lift | Captures 11.9% of readmissions.
  - **Top-10% Capacity:** Precision 23.76% | 2.09x Lift | Captures 20.9% of readmissions.
  - **Top-20% Capacity:** Precision 19.46% | 1.71x Lift | Captures 34.2% of readmissions.
  - **Isotonic Calibration:** Brier score 0.0973 (55.8% drop); ECE 0.39% (<0.4% deviation across all risk deciles).

---

### Slide 8: Clinical Explainability & Algorithmic Fairness
- **Category:** `07. CLINICAL XAI & EQUITY AUDIT`
- **Slide Title:** TreeSHAP Factor Attribution & Demographic Fairness Audit
- **Key Points:**
  - **Local TreeSHAP:** Instant computation (~35ms per request) translated into plain language.
  - **Subgroup Equity:** Stable performance across Caucasian (AUC 0.6540), African American (AUC 0.6720), Hispanic (AUC 0.6657).
  - **Demographic-Blinded Experiment:** Stripping race and gender yielded 0.6597 AUC (vs 0.6590 main model).
  - **Conclusion:** Model predictions are driven by clinical trajectory and comorbidity acuity, not demographic proxies.

---

### Slide 9: Live Clinical Command Center UX
- **Category:** `08. PRODUCTION USER EXPERIENCE`
- **Slide Title:** Actionable Decision Support: Live Discharge Desk & Precedent Orbit
- **Key Points:**
  - **Ranked Discharge Desk (`/v1/worklist`):** Dynamic capacity slider matching nurse staffing.
  - **Precedent Orbit:** Visualizes 20 historical peer encounters with Wilson 95% CI readmission rates.
  - **Deterministic Care Transitions:** Auto-assigns rapid 48h call, pharmacist med recon, and home health warm handoff.
  - **Counterfactual Sensitivity Lab (`/v1/what-if`):** Clinicians simulate interventions to evaluate expected risk reduction.

---

### Slide 10: Production Deployment & Cloud Architecture
- **Category:** `09. LIVE DEPLOYMENT & ROADMAP`
- **Slide Title:** Live Production Deployment, API Contracts & Next Steps
- **Key Points:**
  - **Live Backend:** FastAPI on Render (https://readmitiq-wlw1.onrender.com) with OpenAPI 3.1 specs.
  - **Live Frontend:** React SPA on Vercel (https://readmit-iq.vercel.app).
  - **Footprint:** Lite mode uses 1.69 MB disk and <195 MB RAM.
  - **Real-Time Governance:** PSI drift monitored across all 42 features (PSI < 0.05).
  - **Roadmap:** SMART-on-FHIR R4 integration and 90-day silent shadow hospital pilot.
