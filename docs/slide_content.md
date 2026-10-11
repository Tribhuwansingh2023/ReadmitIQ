# ReadmitIQ Presentation Slide Content (SOA Use Case #6)
## Mentor: Sulagna Tripathy | Track: Supervised Learning + RAG Feature Engineering

Each section below corresponds to one presentation slide for the pitch, containing exact empirical metrics extracted directly from project evaluation artifacts (`reports/metrics/ablation_table.json` and `reports/metrics/test_results.json`).

---

## Slide 1: Title & Executive Value Proposition
- **Category:** `SOA USE CASE #6 | HOSPITAL READMISSION RISK PREDICTION`
- **Title:** ReadmitIQ: Supervised Readmission Prediction via Case-Based RAG Retrieval
- **Subtitle:** Calibrated Risk Triage, Structured Precedent Retrieval & Actionable Care Transitions
- **Key Metrics:**
  - Evaluated on UCI Diabetes 130-US Hospitals (N = 99,340 eligible encounters across 70,442 patients).
  - Primary Model: LightGBM + Structured RAG | Test AUC: 0.6590 [95% CI: 0.6458, 0.6728].
  - Operational Lift: 2.09x Lift at Top-10% Outreach Capacity | ECE: 0.39% (Brier: 0.0973).
  - Scientific Integrity: Paired Bootstrap RAG Delta on test is +0.0004 (CI crosses zero); RAG provides clinical trust.

---

## Slide 2: Clinical Problem Statement & Operational Economics
- **Category:** `01. PROBLEM CONTEXT & HEALTHCARE ECONOMICS`
- **Title:** The $26B Readmission Bottleneck: Why Manual Triage Fails
- **Key Points:**
  - CMS penalties up to 3% of inpatient revenue under HRRP for excess 30-day readmissions.
  - Nurse phone outreach teams can only follow up with 5% to 10% of discharged patients.
  - Clinicians ignore isolated black-box scores like "74% Risk" without clinical precedents.
  - ReadmitIQ aligns risk triage with capacity constraints and case-based evidence.

---

## Slide 3: Real-World Clinical Data & 4 Leakage Traps
- **Category:** `02. DATASET RIGOR & PREPROCESSING`
- **Title:** Real-World Clinical Data: Defeating 4 Critical Leakage Traps
- **Key Points:**
  - UCI Diabetes 130-US Hospitals: 101,766 raw encounters -> 99,340 eligible encounters.
  - **Trap 1: Multi-Encounter Patient Leak:** GroupKFold on `patient_nbr` ensuring 0% patient overlap across Train (69,535), Val (14,902), Test (14,903).
  - **Trap 2: Mortality & Hospice Contamination:** Filtered 2,426 expired/hospice records.
  - **Trap 3: Informative Missingness:** Preserved "None" as an explicit state for unmeasured HbA1c/glucose.
  - **Trap 4: Categorical ICD Explosion:** Collapsed 700+ codes into 9 clinical organ systems.

---

## Slide 4: Synthetic Clinical Note Summaries & RAG Architecture
- **Category:** `03. RAG FEATURE ENGINEERING`
- **Title:** Case Precedent Retrieval: From Note Summaries to Engineered RAG Features
- **Key Points:**
  - **Synthetic Note Generator:** Audited clinical narratives with zero target label tokens.
  - **Latent Embedding Space:** 64-dimensional TruncatedSVD projection fit strictly on Train.
  - **Historical Case Retrieval:** FAISS IndexFlatIP indexing 69,535 training encounters (<2ms search).
  - **OOF Target Masking:** 5-fold cross-validation on Train prevents self-retrieval label leakage.
  - **Engineered Feature (`rag_rate_k20`):** Empirical 30-day readmission rate among top-20 historical peers.

---

## Slide 5: Comprehensive Ablation Benchmark Table
- **Category:** `04. BENCHMARKING & METRIC ABLATION`
- **Title:** Model Benchmark Across AUC-ROC, F1, Precision & Recall

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

## Slide 6: Statistical Hypothesis Testing & Scientific Integrity
- **Category:** `05. HYPOTHESIS TESTING & STATISTICAL RIGOR`
- **Title:** Statistical Hypothesis Testing: What Does RAG Really Deliver?
- **Key Points:**
  - 1,000-sample paired bootstrap test on held-out test split: Delta AUC = +0.0004 [95% CI: -0.0025, +0.0031].
  - Confidence interval crosses zero (p > 0.05).
  - RAG does not inflate tabular ROC-AUC; its real value is **interpretable case-based reasoning and clinician trust**.

---

## Slide 7: Locked Test Evaluation & Capacity-Aware Triage
- **Category:** `06. INDEPENDENT TEST EVALUATION`
- **Title:** Held-Out Test Results: Calibration & Operational Lift
- **Key Points:**
  - Evaluated on N = 14,903 locked test encounters: AUC-ROC 0.6590 [95% CI: 0.6458, 0.6728].
  - Top-10% Outreach Workload: Precision 23.76% | Lift 2.09x | Recall 20.9%.
  - Isotonic calibration reduces Brier score by 55.8% to 0.0973; ECE drops to 0.39%.

---

## Slide 8: Clinical TreeSHAP & Algorithmic Fairness Audit
- **Category:** `07. CLINICAL XAI & EQUITY AUDIT`
- **Title:** TreeSHAP Factor Attribution & Demographic Fairness Audit
- **Key Points:**
  - Local TreeSHAP attributions in ~35ms translated into plain clinical language.
  - Subgroup audit across Caucasian (AUC 0.6540), African American (AUC 0.6720), Hispanic (AUC 0.6657).
  - Demographic-blinded experiment (race and gender removed): Test AUC 0.6597 (vs 0.6590 main model).
  - Predictions are driven by illness severity and utilization history, not demographic proxies.

---

## Slide 9: Live Clinical Command Center UX
- **Category:** `08. PRODUCTION USER EXPERIENCE`
- **Title:** Actionable Decision Support: Live Discharge Desk & Precedent Orbit
- **Key Points:**
  - Dynamic capacity slider (1% to 30%) on the live worklist matching nurse bandwidth.
  - Precedent Orbit visualizes 20 historical peers with Wilson 95% CI readmission rates.
  - Deterministic transition care protocols (phone calls, med recon, home health).
  - What-If Sensitivity Lab for clinical counterfactual exploration.

---

## Slide 10: Production Deployment & Cloud Architecture
- **Category:** `09. LIVE DEPLOYMENT & ROADMAP`
- **Title:** Live Production Deployment, API Contracts & Next Steps
- **Key Points:**
  - Live Backend: FastAPI on Render (https://readmitiq-wlw1.onrender.com) with OpenAPI 3.1.
  - Live Frontend: React/Vite on Vercel (https://readmit-iq.vercel.app).
  - Serving Footprint: Lite mode uses 1.69 MB disk and <195 MB RAM.
  - Real-Time PSI Governance: Drift monitored across all 42 features (PSI < 0.05).
  - Roadmap: SMART-on-FHIR R4 integration and 90-day silent shadow hospital pilot.
