# ReadmitIQ Signature Differentiators

ReadmitIQ introduces two unique differentiators that elevate it beyond static risk dashboards into a safety-critical clinical decision support platform.

---

## Differentiator 1: What-If Risk Sensitivity Explorer

### Purpose
Care transition teams need to explore how hypothetical changes in patient care plans (such as reducing acute length of stay, optimizing medication regimens, or establishing outpatient follow-up) would impact the model's risk score.

### Interface & Operation
- **Side-by-Side Comparison:** Compares the actual encounter baseline against a simulated modified care profile.
- **Real-Time Pipeline Scoring:** Evaluates modified profiles through the actual, unmocked LightGBM + Isotonic serving pipeline (`POST /v1/what-if`).
- **Delta Analysis:** Displays absolute score changes, calibrated probability shifts, relative risk percentage change, and serving tier transitions.
- **Top Driver Re-computation:** Dynamically recalibrates SHAP log-odds explanations for the modified encounter.

### Clinical & Causal Disclaimer
> **IMPORTANT REGULATORY GUARDRAIL:**
> The What-If Risk Explorer performs **associative model sensitivity analysis**, *not* a causal treatment effect estimate.
> Altering a feature value demonstrates how the frozen statistical model responds; it does **not** guarantee that an intervention will reduce actual clinical readmission risk in a given patient. Clinical decision-making must remain with licensed clinicians.

---

## Differentiator 2: Trust & Safety Surveillance Monitor

### Purpose
Clinical machine learning models frequently suffer from silent performance degradation due to covariate shift, population drift, data quality decay, or unpowered subgroup claims. The Trust & Safety Monitor provides continuous real-time surveillance.

### Surveillance Modules

#### 1. Score & Covariate Drift (Population Stability Index - PSI)
- **Score Distribution PSI:** Compares recent live raw score deciles against frozen training baseline deciles.
  - $\text{PSI} < 0.10$: Minimal shift (Stable).
  - $0.10 \le \text{PSI} < 0.25$: Moderate shift (Warning).
  - $\text{PSI} \ge 0.25$: High shift (Actionable Alert).
- **Feature Covariate PSI:** Evaluates individual feature drift across clinical predictors (e.g., inpatient admissions, primary diagnosis, HbA1c testing rates).

#### 2. RAG Vector Store Health
- Tracks FAISS index availability, vector count ($n = 69,535$), embedding dimension (64-d SVD), and logged query retrieval failures.
- Enforces graceful degradation disclosures if vector retrieval fails.

#### 3. Data Quality & Rejection Logging
- Tracks schema conformity, missing feature rates, and rejected API/CSV payloads in `data_quality_events`.
- Preserves every rejected row or formatting error for auditing.

#### 4. Small-n Statistical Reliability Guardrails
- **The Problem:** Subgroup audits or live discrimination metrics computed on small sample sizes ($n < 50$ or $n < 200$) yield unstable point estimates with wide, uninformative confidence intervals.
- **The ReadmitIQ Guardrail:** Live empirical ROC-AUC and subgroup benchmark claims are explicitly withheld unless sample sizes satisfy minimum statistical power criteria ($n \ge 200$ resolved outcomes and $\ge 20$ positive readmissions). Subgroups with $n < 100$ encounters are visibly marked with an **Unpowered Small-n Guard** to prevent misleading compliance conclusions.
