# ReadmitIQ API Specification & Contract

## 1. Overview
The ReadmitIQ REST API delivers real-time clinical readmission risk inference, patient retrieval, SHAP explainability, and retrospective cohort replay. Built on FastAPI and Pydantic, the service complies strictly with the frozen model selection (`models/FROZEN_SELECTION.json`) and loads precomputed evaluation artifacts for monitoring transparency.

- **Base URL**: `http://localhost:8000` (or `API_BASE_URL` env variable)
- **API Version**: `v1.0.0`
- **Model Runtime**: LightGBM Classifier with Structured FAISS RAG Retriever & Isotonic Calibration
- **CORS Support**: Configurable via `CORS_ORIGINS` environment variable (default: `http://localhost:3000,http://localhost:5173`)
- **Audit Logging**: Prediction requests logged to `api/logs/predictions.jsonl` (anonymized clinical attributes only, zero external identifiers).

---

## 2. Endpoints

### 2.1 `GET /health`
Returns system health, git commit hash, model freeze timestamp, and status of all serialized serving artifacts.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "git_commit_hash": "cb85b01...",
  "frozen_timestamp": "2026-10-08T22:25:00Z",
  "model_name": "LightGBM + Structured RAG",
  "artifacts_loaded": {
    "model": true,
    "preprocessor": true,
    "calibrator": true,
    "retriever_faiss": true,
    "retriever_meta": true,
    "feature_list": true
  }
}
```

---

### 2.2 `POST /predict`
Performs end-to-end inference on a single patient encounter: retrieves similar historical encounters from the FAISS train index, extracts the structured neighborhood readmission rate, predicts calibrated risk probability, assigns risk tier based on frozen capacity thresholds, and generates plain-language SHAP explanations.

#### Request Body Schema
```json
{
  "age": "[60-70)",
  "gender": "Female",
  "race": "Caucasian",
  "time_in_hospital": 4,
  "number_inpatient": 2,
  "number_emergency": 0,
  "number_outpatient": 1,
  "num_medications": 14,
  "num_lab_procedures": 45,
  "num_procedures": 1,
  "number_diagnoses": 8,
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
  "insulin": "Down",
  "metformin": "Steady"
}
```

#### Response `200 OK`
```json
{
  "encounter_id": "SIM_1001",
  "calibrated_risk_probability": 0.2642,
  "risk_tier": "Critical",
  "decision_threshold": 0.1495,
  "predicted_readmission": true,
  "capacity_thresholds": {
    "top_5pct_critical": 0.2472,
    "top_10pct_high": 0.1940,
    "top_20pct_medium": 0.1551
  },
  "top_risk_increasing_factors": [
    {
      "feature": "number_inpatient",
      "value": 2,
      "shap_value": 0.384,
      "explanation": "2 prior inpatient admissions in the past year raises risk"
    },
    {
      "feature": "A1Cresult",
      "value": ">8",
      "shap_value": 0.192,
      "explanation": "Elevated HbA1c (>8%) raises risk"
    }
  ],
  "top_risk_reducing_factors": [
    {
      "feature": "discharge_disposition_group",
      "value": "Home",
      "shap_value": -0.215,
      "explanation": "Routine discharge directly to home lowers risk"
    }
  ],
  "similar_patients_summary": {
    "k_neighbors": 20,
    "neighbor_readmission_rate": 0.30,
    "top_neighbors": [
      {
        "encounter_id": 8472910,
        "similarity_score": 0.942,
        "age": "[60-70)",
        "time_in_hospital": 4,
        "primary_diagnosis": "Circulatory",
        "prior_inpatient": 2,
        "readmitted_30d": 1
      }
    ]
  }
}
```

---

### 2.3 `POST /similar`
Retrieves top-$k$ nearest historical training encounters for a query patient profile using the exact Structured FAISS cosine similarity index.

#### Request Body Schema
```json
{
  "patient_features": { ... },
  "k": 20
}
```

#### Response `200 OK`
```json
{
  "k_requested": 20,
  "neighbor_readmission_rate": 0.30,
  "similar_encounters": [
    {
      "encounter_id": 8472910,
      "similarity_score": 0.942,
      "age": "[60-70)",
      "gender": "Female",
      "time_in_hospital": 4,
      "primary_diagnosis": "Circulatory",
      "prior_inpatient": 2,
      "readmitted_30d": 1
    }
  ]
}
```

---

### 2.4 `POST /explain`
Returns fine-grained SHAP attribution values and narrative clinical explanations for a patient.

#### Response `200 OK`
```json
{
  "top_risk_increasing": [ ... ],
  "top_risk_reducing": [ ... ],
  "all_attributions": {
    "number_inpatient": 0.384,
    "A1Cresult": 0.192, ...
  }
}
```

---

### 2.5 `GET /cohort?capacity=0.10`
Performs dynamic capacity-constrained cohort replay over the held-out TEST split (14,903 encounters) using precomputed calibrated probabilities from `reports/demo/cohort_scored.parquet`.

- **Parameters**: `capacity` (float, range `0.01` to `0.50`, default `0.10`)

#### Response `200 OK`
```json
{
  "mode": "Retrospective Replay (Held-Out Test Split)",
  "capacity_fraction": 0.10,
  "total_test_encounters": 14903,
  "flagged_encounters_count": 1490,
  "total_true_readmissions_in_cohort": 1698,
  "true_readmissions_caught": 354,
  "precision": 0.2376,
  "recall": 0.2085,
  "lift_vs_random": 2.09,
  "threshold_at_capacity": 0.1940,
  "top_patients": [
    {
      "encounter_id": 12849102,
      "patient_nbr": 829410,
      "risk_score": 0.4851,
      "risk_tier": "Critical",
      "readmitted_30d": 1,
      "age": "[70-80)",
      "time_in_hospital": 8,
      "number_inpatient": 4,
      "diag_1_group": "Circulatory"
    }
  ]
}
```

---

### 2.6 `GET /metrics`
Reads and serves verified offline evaluation artifacts without calculating or hardcoding numbers inside the API runtime:
- Validation Ablation Table (`reports/metrics/ablation_table.json`)
- Final Test Results (`reports/metrics/test_results.json`)
- Subgroup Fairness Audit (`reports/metrics/fairness_audit.json`)
- Population Stability Index Drift Monitoring (`reports/metrics/drift_metrics.json`)

---

## 3. v1 Clinical Workflow & Live Monitoring Endpoints

All `v1` responses include the mandatory header:
```http
X-Model-Version: 1.0.0 (LightGBM + Structured RAG)
```

### 3.1 `POST /v1/discharges`
Live clinical ingestion for single discharges or vectorized batches (up to 5,000 encounters). Scored in < 2.0s per 1,000 rows.

#### Request (Single Encounter)
```json
{
  "external_ref": "ENC-2026-9021",
  "age": "[65-75)",
  "gender": "Female",
  "race": "Caucasian",
  "time_in_hospital": 5,
  "number_inpatient": 2,
  "num_medications": 18,
  "num_lab_procedures": 52,
  "diag_1_group": "Circulatory",
  "A1Cresult": ">8",
  "diabetesMed": "Yes",
  "change": "Ch",
  "insulin": "Up"
}
```

#### Response `201 Created`
```json
{
  "id": 142,
  "received_at": "2026-10-10T01:30:00Z",
  "source": "api",
  "external_ref": "ENC-2026-9021",
  "raw_score": 0.694218,
  "calibrated_prob": 0.2814,
  "tier": "Critical",
  "status": "new",
  "model_version": "LightGBM + Structured RAG",
  "neighbor_summary": {
    "k_neighbors": 20,
    "neighbor_readmission_rate": 0.3500
  }
}
```

---

### 3.2 `POST /v1/discharges/upload` & `GET /v1/templates/discharge.csv`
- `GET /v1/templates/discharge.csv`: Downloads standard CSV template with headers and 3 clinical example encounters.
- `POST /v1/discharges/upload`: Uploads CSV file (multipart/form-data). Returns validation summary report with row-level error breakdown.

#### Upload Response `200 OK`
```json
{
  "filename": "shift_handoff_20261010.csv",
  "total_rows_received": 120,
  "accepted_count": 118,
  "rejected_count": 2,
  "accepted_rows": [ ... ],
  "rejected_rows": [
    {
      "row_index": 14,
      "error": "time_in_hospital: Input should be greater than or equal to 1",
      "external_ref": "ROW_14"
    }
  ]
}
```

---

### 3.3 `GET /v1/worklist`
Care-team worklist sorted strictly by RAW model score (tie-break by ID).

#### Query Parameters
- `status`: Filter by status (`new`, `assigned`, `contacted`, `followup_scheduled`, `closed`, or `open`)
- `tier`: Filter by risk tier (`Critical`, `High`, `Medium`, `Low`)
- `search`: Full text search matching external ref, ID, or clinical features
- `capacity`: Integer N (e.g. `capacity=50`). Returns the top N open items by raw score for constrained intervention teams.
- `page`: Page index (1-based)
- `page_size`: Encounters per page (default 50)

#### Response `200 OK`
```json
{
  "mode": "standard",
  "page": 1,
  "page_size": 50,
  "total_count": 340,
  "tier_counts": {
    "Critical": 22,
    "High": 31,
    "Medium": 58,
    "Low": 229
  },
  "items": [
    {
      "id": 142,
      "received_at": "2026-10-10T01:30:00Z",
      "source": "api",
      "external_ref": "ENC-2026-9021",
      "raw_score": 0.694218,
      "calibrated_prob": 0.2814,
      "tier": "Critical",
      "status": "new",
      "assignee": null,
      "updated_at": "2026-10-10T01:30:00Z",
      "outcome_readmitted_30d": null,
      "patient_summary": {
        "age": "[65-75)",
        "gender": "Female",
        "time_in_hospital": 5,
        "primary_diagnosis": "Circulatory",
        "number_inpatient": 2
      }
    }
  ]
}
```

---

### 3.4 `GET /v1/discharges/{id}`
Returns clinical features, raw & calibrated scores, tier, TreeSHAP attributions computed on demand and cached, nearest historical train neighbors, deterministic intervention recommendations, action history, and audit trail.

---

### 3.5 Clinical Workflow Mutations
- `PATCH /v1/discharges/{id}`: Update status, assignee, or clinician tier override. If `clinician_override_tier` is provided, `override_reason` is strictly required. Writes an audit entry.
- `POST /v1/discharges/{id}/actions`: Records clinical touchpoint (`type`, `note`, `actor`). Stored free-text carries a "do not enter PHI" warning. Writes an audit entry.
- `POST /v1/discharges/{id}/outcome`: Records verified 30-day readmission outcome (`0` or `1`). Writes an audit entry.

---

### 3.6 Simulated Feed & Time Machine
- `POST /v1/feed/start`: `{ "rate_per_minute": 60.0, "seed": 42 }`. Replays held-out test encounters with labels stripped. Ground truth sealed in `feed_truth`.
- `POST /v1/feed/stop`: Halts simulated feed loop.
- `GET /v1/feed/status`: Telemetry including simulated clock, emission rate, and total emitted.
- `POST /v1/feed/time-machine`: `{ "days": 30 }`. Advances simulated clock by N days and releases outcomes from `feed_truth` for discharges older than 30 days. Sealed truth is never accessible prior to release.

---

### 3.7 `GET /v1/monitoring`
Live operational monitoring dashboard:
- Tier mix vs. validation baseline (Critical 5%, High 5%, Medium 10%, Low 80%) with mix shift alerts.
- Score-drift PSI comparing live raw scores to training deciles from `serving_thresholds.json`.
- Input-feature PSI using `src/drift.py`.
- Data quality validation rates.
- Outcomes evaluation: Guarded under small sample size (<200 resolved outcomes or <20 positives displays explicit `"insufficient_outcomes"` state). When sufficient, computes live AUC with 95% bootstrap CI, precision at capacity, and calibration decile curve.

---

### 3.8 `GET /v1/model/info`
Model provenance metadata, Git commit hash, serving raw-score cutoffs, training/validation/test split summary, and link to model card.

---

## 4. Error Contract
Standardized RFC-7807 compliant error format:
```json
{
  "error": "Validation Error",
  "detail": [
    {
      "field": "time_in_hospital",
      "message": "Input should be greater than or equal to 1"
    }
  ],
  "status_code": 422
}
```

