# ReadmitIQ System Architecture

## Overview
ReadmitIQ is a clinical decision support system designed for 30-day all-cause hospital readmission risk prediction and evidence-based care coordination. Unlike conventional machine learning dashboards that stop at binary risk scoring, ReadmitIQ integrates prediction, precedent case evidence, automated care-transition task scheduling, capacity-aware triage, and closed-loop follow-up tracking into a unified clinical workflow: the **ReadmitIQ ActionLoop**.

```
+---------------------------------------------------------------------------------------------------+
|                                      READMITIQ ARCHITECTURE                                       |
+---------------------------------------------------------------------------------------------------+

     +-----------------------+        +------------------------+        +----------------------+
     |  EHR Ingestion (CSV)  |        |  Real-Time REST API    |        |  Simulated Feed Replay|
     +-----------+-----------+        +-----------+------------+        +----------+-----------+
                 |                                |                                |
                 +--------------------------------+--------------------------------+
                                                  |
                                                  v
                              +---------------------------------------+
                              |      FastAPI Clinical Gateway (v1)     |
                              |  - Pydantic Schema Validation         |
                              |  - Data Quality & Rejection Logging   |
                              |  - Mandatory X-Model-Version Header   |
                              +-------------------+-------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
           +---------------------------+                     +---------------------------+
           |     Predictive Core       |                     |     RAG Vector Engine     |
           |  - Tabular Preprocessor   |                     |  - Base Dense Transformer |
           |  - Frozen LightGBM (42f)  |                     |  - Truncated SVD (64-dim) |
           |  - Isotonic Calibrator    |                     |  - L2 Normalized FAISS    |
           |  - Decile Serving Tiers   |                     |  - 69k Reference Vectors  |
           +-------------+-------------+                     +-------------+-------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                              +---------------------------------------+
                              |    ReadmitIQ ActionLoop Coordinator   |
                              |  1. Risk Intelligence (Raw + Calib)   |
                              |  2. Precedent Evidence (k=20 peers)   |
                              |  3. Action Plan (Transition Tasks)    |
                              |  4. Capacity-Aware Triage Queue       |
                              |  5. Closed-Loop Follow-up & Audit     |
                              +-------------------+-------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
           +---------------------------+                     +---------------------------+
           |   Persistence (SQLite)    |                     |   React / Vite Web UI     |
           |  - Encounters & Features  |                     |  - Discharge Desk (Queue) |
           |  - Care Tasks & Statuses  |                     |  - Patient Brief Modal    |
           |  - Touchpoint Audit Trail |                     |  - What-If Risk Explorer  |
           |  - Sealed Ground-Truth    |                     |  - Trust & Safety Monitor |
           +---------------------------+                     +---------------------------+
```

---

## Component Specifications

### 1. Tabular Machine Learning Pipeline
- **Classifier:** LightGBM Gradient Boosted Decision Trees trained on 42 clinical encounter covariates.
- **Model Parameters:** Fixed in `models/final/model.joblib`.
- **Calibration:** Non-parametric Isotonic Regression (`models/final/calibrator.joblib`) mapping raw scores to empirical probabilities.
- **Serving Thresholds:**
  - **Critical Tier (Top 5%):** $\text{Raw Score} \ge 0.6839$
  - **High Tier (Top 10%):** $\text{Raw Score} \ge 0.6408$
  - **Medium Tier (Top 20%):** $\text{Raw Score} \ge 0.5793$
  - **Low Tier (Remaining 80%):** $\text{Raw Score} < 0.5793$

### 2. Retrieval-Augmented Generation (RAG) Vector Store
- **Index Type:** `faiss.IndexFlatIP` (inner product over unit-normalized embeddings, equivalent to cosine similarity).
- **Latent Dimension:** 64-dimensional clinical vector space generated via truncated Singular Value Decomposition (`models/final/structured_svd.joblib`).
- **Reference Cohort:** 69,535 patient-disjoint training encounters (`models/final/retriever_meta.parquet`).
- **Retrieval Scale:** $k = 20$ nearest neighbors retrieved per patient encounter.

### 3. SQLite Storage Engine
- **Mode:** Write-Ahead Logging (`WAL`) for concurrent reader/writer support.
- **Schema:**
  - `discharges`: Clinical encounters, features JSON, raw score, calibrated probability, serving tier, outreach status, and sealed outcome.
  - `tasks`: Action plan checklist items with title, priority, assignee, status (`pending`, `in_progress`, `completed`), and completion timestamp.
  - `discharge_actions`: Coordinator contact logs (phone calls, home visits, med rec).
  - `audit_log`: System lifecycle audit trail.
  - `data_quality_events`: Validation failures, rejected payloads, and schema conformity anomalies.
