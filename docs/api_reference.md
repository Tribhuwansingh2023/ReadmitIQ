# ReadmitIQ REST API Reference

All requests and responses use JSON format. Every HTTP response attaches the mandatory header:
`X-Model-Version: 1.0.0 (LightGBM + Structured RAG)`

---

## 1. System & Health Endpoints

### `GET /health`
Returns system status, serving mode, git commit hash, and loaded runtime artifacts.
- **Response:**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "serve_mode": "full",
    "git_commit_hash": "b2f3b54",
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

### `GET /v1/model/info`
Returns frozen model metadata, training commit, capacity thresholds, and split counts.

---

## 2. Risk Inference Endpoints

### `POST /v1/predict` (and `POST /predict`)
Scores a patient encounter through the LightGBM + Isotonic pipeline and computes TreeSHAP explanations.
- **Request Body:** 42 clinical encounter covariates (see Data Dictionary).
- **Response:**
  ```json
  {
    "raw_score": 0.544541,
    "calibrated_prob": 0.1471,
    "tier": "Low",
    "decision_threshold": 0.1245,
    "predicted_readmission": true,
    "shap_explanation": {
      "top_risk_increasing": [...],
      "top_risk_reducing": [...]
    }
  }
  ```

---

## 3. Encounter Intake & Worklist

### `POST /v1/discharges`
Ingests a single patient encounter or batch (up to 5,000) with automatic scoring, RAG precedent retrieval, and SQLite persistence.
- **Status:** `201 Created`

### `GET /v1/worklist`
Returns active discharge worklist paged and ranked by `raw_score DESC, received_at DESC, id ASC`.
- **Query Parameters:** `page`, `page_size`, `tier`, `status`, `search`, `capacity_limit`.

### `GET /v1/discharges/{id}`
Returns full clinical brief: features, SHAP factors, 20 RAG precedent neighbors, decision support suggestions, care transition tasks, and chronological timeline.

### `PATCH /v1/discharges/{id}`
Updates outreach workflow status (`new`, `assigned`, `contacted`, `followup_scheduled`, `closed`), assignee, or applies a clinical override with mandatory rationale.

---

## 4. Care Transition Action Plan

### `GET /v1/discharges/{id}/tasks`
Lists all follow-up care transition tasks for an encounter.

### `POST /v1/discharges/{id}/tasks`
Creates a new actionable care transition task.
- **Fields:** `title`, `category`, `priority` (`High`, `Medium`, `Low`), `owner`, `rationale`, `due_date`.

### `PATCH /v1/discharges/{id}/tasks/{task_id}`
Updates task status (`pending`, `in_progress`, `completed`, `cancelled`), assignee, or records completion with actor and completion note.

---

## 5. Touchpoints & Closed-Loop Follow-up

### `POST /v1/discharges/{id}/actions`
Logs an outreach touchpoint (phone call, home visit, med rec, transportation).

### `POST /v1/discharges/{id}/outcome`
Records verified post-discharge 30-day readmission outcome ($0$ or $1$) with clinician notes.

---

## 6. What-If Risk Sensitivity Explorer

### `POST /v1/what-if`
Simulates model sensitivity under hypothetical input feature variations without mutating patient records.
- **Request:**
  ```json
  {
    "discharge_id": 1,
    "modified_features": {
      "time_in_hospital": 2,
      "num_medications": 8
    }
  }
  ```
- **Response:** Side-by-side baseline vs modified scores, probability change, relative risk delta, and tier transition.

---

## 7. Operational Surveillance & Drift Monitoring

### `GET /v1/monitoring`
Returns live operational metrics: total counts, tier mix, raw-score distribution PSI, feature covariate PSI, RAG vector store health, data quality rejection events, and empirical ROC-AUC with small-n statistical reliability guards.
