# ReadmitIQ ActionLoop Specification

The **ReadmitIQ ActionLoop** is the core operational framework of ReadmitIQ. It bridges predictive machine learning and frontline clinical discharge coordination through five interconnected stages.

```
       +-----------------------------------------------------------+
       |                  READMITIQ ACTIONLOOP                     |
       +-----------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
   [ 1. Risk Intelligence ]                    [ 2. Precedent Evidence ]
   Raw Score + Calibrated Prob                 k=20 Structurally Similar Peers
   Decile Cutoffs & SHAP Drivers               Neighborhood Rate & Wilson 95% CI
               |                                           |
               +---------------------+---------------------+
                                     |
                                     v
                       [ 3. Care Action Plan ]
                       Automated Driver-Triggered Tasks
                       Status: Pending -> In-Progress -> Completed
                                     |
                                     v
                     [ 4. Capacity-Aware Triage ]
                     Deterministic Rank: Raw Score DESC -> Arrival DESC
                     Top-N Outreach Allocation & Dynamic Recalculation
                                     |
                                     v
                   [ 5. Closed-Loop Follow-up ]
                   Chronological Audit Trail & Touchpoints
                   Sealed Ground-Truth Post-Discharge Surveillance
```

---

## Stage 1: Risk Intelligence
- **Input:** 42 tabular clinical encounter covariates.
- **Inference:** The frozen LightGBM model outputs a continuous raw score $\in [0, 1]$.
- **Calibration:** Non-parametric isotonic regression transforms the raw score into a calibrated 30-day readmission probability $P(\text{readmit} \mid X)$.
- **Explainability:** Fast TreeSHAP calculates exact additive log-odds feature contributions for both top risk-increasing and top protective factors.
- **Serving Tier Allocation:**
  - Critical Tier (Top 5% cutoff)
  - High Tier (Top 10% cutoff)
  - Medium Tier (Top 20% cutoff)
  - Low Tier (Remaining 80%)

---

## Stage 2: Precedent Evidence
- **Mechanism:** Vectorized nearest-neighbor search across a 64-dimensional clinical vector space using `faiss.IndexFlatIP`.
- **Peer Cohort:** 20 structurally similar historical encounters from the patient-disjoint training split ($n = 69,535$).
- **Neighborhood Statistics:** Readmission rate across the 20 neighbors with Wilson score 95% confidence intervals.
- **Scientific Guardrail:** Retrieval of similar historical precedents provides clinical context and explainability to human care coordinators. It does **not** add independent predictive ROC-AUC lift over the tabular gradient booster (Paired $\Delta\text{AUC} = 0.0000$).

---

## Stage 3: Care Transition Action Plan
- **Automatic Task Generation:** Specific SHAP feature triggers automatically seed protocolized transition tasks upon encounter intake:
  - High inpatient count $\rightarrow$ Post-discharge home health or telehealth outreach within 48h.
  - High medication count $\rightarrow$ Pharmacist medication reconciliation and reconciliation checklist.
  - Elevated HbA1c / uncontrolled glycemia $\rightarrow$ Outpatient diabetes education and CGM follow-up.
  - Social or transportation barriers $\rightarrow$ Case management transportation coordination.
- **Lifecycle Management:** Tasks transition through discrete states: `pending` $\rightarrow$ `in_progress` $\rightarrow$ `completed` (or `cancelled`).
- **Auditability:** Completion requires actor identification and an optional clinical completion note.

---

## Stage 4: Capacity-Aware Priority Triage
- **Deterministic Ordering:** Worklist queues are ranked strictly by:
  1. `raw_score DESC` (eliminates isotonic calibration plateaus where multiple patients share identical rounded probabilities)
  2. `received_at DESC` (deterministic arrival-time tie-breaking)
  3. `id ASC` (guaranteed total order)
- **Capacity Thresholding:** Hospital coordinators configure outreach capacity $N$ (e.g., top 15 patients per shift).
- **Dynamic Recalculation:** As higher-priority encounters are contacted or closed, subsequent encounters dynamically advance into outreach capacity.

---

## Stage 5: Closed-Loop Follow-up & Surveillance
- **Touchpoint History:** Logs phone outreach, home visits, medication reviews, and clinical override events.
- **Outcome Verification:** Sealed ground-truth post-discharge outcomes ($0 = \text{no readmission}$, $1 = \text{readmitted}$) are recorded via registry or Time Machine simulation.
- **Chronological Timeline:** Every intake, task mutation, outreach contact, and sealed outcome is unified into a single chronological audit trail.
