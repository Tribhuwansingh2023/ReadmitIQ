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

## 3. Error Contract
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
