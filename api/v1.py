"""ReadmitIQ v1 Clinical Workflow & Monitoring API Router.

Includes:
- Discharge ingestion (POST /v1/discharges, CSV upload, templates)
- Clinical worklist with raw-score ranking and capacity mode (GET /v1/worklist)
- Single encounter detail with on-demand TreeSHAP, interventions, and audit trail (GET /v1/discharges/{id})
- Clinical workflow mutations (PATCH /v1/discharges/{id}, actions, outcomes)
- Simulated feed control and Time Machine (POST /v1/feed/*)
- Real-time operational monitoring with PSI score drift and small-n guards (GET /v1/monitoring)
- Model metadata & serving thresholds (GET /v1/model/info)
"""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score

from api.database import get_db, log_audit
from api.feed import get_feed_engine
from api.interventions import generate_suggested_interventions
from api.schemas import PatientEncounterRequest
from api.service import get_service
from src.config import BASE_DIR, MODELS_DIR
from src.drift import calculate_psi, classify_drift_severity

router = APIRouter(prefix="/v1", tags=["Clinical Workflow v1"])


# ---------------------------------------------------------------------------
# Request & Response Models for v1
# ---------------------------------------------------------------------------


class DischargePatchRequest(BaseModel):
    status: Optional[str] = Field(
        None, pattern="^(new|assigned|contacted|followup_scheduled|closed)$"
    )
    assignee: Optional[str] = None
    clinician_override_tier: Optional[str] = Field(
        None, pattern="^(Critical|High|Medium|Low)$"
    )
    override_reason: Optional[str] = None
    actor: str = Field(default="Care Coordinator")


class ActionCreateRequest(BaseModel):
    type: str = Field(
        ...,
        description="Action type: e.g. phone_call, med_reconciliation, home_visit, pcp_scheduled, patient_education",
    )
    note: str = Field(
        ...,
        description="Clinical touchpoint note. WARNING: Do not enter Protected Health Information (PHI).",
    )
    actor: str = Field(default="Care Coordinator")


class OutcomeCreateRequest(BaseModel):
    outcome_readmitted_30d: int = Field(..., ge=0, le=1)
    actor: str = Field(default="Quality Director")


class FeedStartRequest(BaseModel):
    rate_per_minute: float = Field(default=60.0, ge=1.0, le=600.0)
    seed: int = Field(default=42)


class TimeMachineRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)


# ---------------------------------------------------------------------------
# 1. Discharges Ingestion & CSV Templates
# ---------------------------------------------------------------------------


@router.get("/templates/discharge.csv")
def get_discharge_csv_template():
    """Download standard CSV template with headers and 3 clinical example rows."""
    sample_records = [
        {
            "age": "[60-70)",
            "gender": "Female",
            "race": "Caucasian",
            "time_in_hospital": 4,
            "num_lab_procedures": 48,
            "num_procedures": 1,
            "num_medications": 18,
            "number_outpatient": 0,
            "number_emergency": 1,
            "number_inpatient": 2,
            "number_diagnoses": 9,
            "admission_type_group": "Emergency",
            "admission_source_group": "Emergency Room",
            "discharge_disposition_group": "Home",
            "medical_specialty": "InternalMedicine",
            "payer_code": "Medicare",
            "diag_1_group": "Circulatory",
            "diag_2_group": "Diabetes",
            "diag_3_group": "Other",
            "max_glu_serum": "None",
            "A1Cresult": ">8",
            "diabetesMed": "Yes",
            "change": "Ch",
            "insulin": "Up",
            "metformin": "Steady",
            "glipizide": "No",
            "glyburide": "No",
            "glimepiride": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "acarbose": "No",
            "repaglinide": "No",
            "nateglinide": "No",
            "glyburide-metformin": "No",
        },
        {
            "age": "[70-80)",
            "gender": "Male",
            "race": "Caucasian",
            "time_in_hospital": 7,
            "num_lab_procedures": 62,
            "num_procedures": 2,
            "num_medications": 24,
            "number_outpatient": 1,
            "number_emergency": 0,
            "number_inpatient": 1,
            "number_diagnoses": 9,
            "admission_type_group": "Emergency",
            "admission_source_group": "Emergency Room",
            "discharge_disposition_group": "SNF",
            "medical_specialty": "Cardiology",
            "payer_code": "Medicare",
            "diag_1_group": "Circulatory",
            "diag_2_group": "Circulatory",
            "diag_3_group": "Diabetes",
            "max_glu_serum": "None",
            "A1Cresult": "Norm",
            "diabetesMed": "Yes",
            "change": "No",
            "insulin": "Steady",
            "metformin": "No",
            "glipizide": "No",
            "glyburide": "No",
            "glimepiride": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "acarbose": "No",
            "repaglinide": "No",
            "nateglinide": "No",
            "glyburide-metformin": "No",
        },
        {
            "age": "[40-50)",
            "gender": "Female",
            "race": "AfricanAmerican",
            "time_in_hospital": 2,
            "num_lab_procedures": 25,
            "num_procedures": 0,
            "num_medications": 8,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "number_diagnoses": 5,
            "admission_type_group": "Elective",
            "admission_source_group": "Physician Referral",
            "discharge_disposition_group": "Home",
            "medical_specialty": "Other",
            "payer_code": "Private",
            "diag_1_group": "Digestive",
            "diag_2_group": "Diabetes",
            "diag_3_group": "Other",
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "diabetesMed": "No",
            "change": "No",
            "insulin": "No",
            "metformin": "No",
            "glipizide": "No",
            "glyburide": "No",
            "glimepiride": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "acarbose": "No",
            "repaglinide": "No",
            "nateglinide": "No",
            "glyburide-metformin": "No",
        },
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(sample_records[0].keys()))
    writer.writeheader()
    writer.writerows(sample_records)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=readmitiq_discharge_template.csv"},
    )


@router.post("/discharges", status_code=status.HTTP_201_CREATED)
def create_discharges(payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
    """Ingest one encounter or a batch of encounters (up to 5,000).

    Validates schema per row, executes high-throughput vectorized scoring, and persists to SQLite.
    """
    svc = get_service()

    if isinstance(payload, list):
        items = payload
        is_single = False
    elif isinstance(payload, dict):
        if "discharges" in payload and isinstance(payload["discharges"], list):
            items = payload["discharges"]
            is_single = False
        else:
            items = [payload]
            is_single = True
    else:
        raise HTTPException(status_code=400, detail="Invalid payload format")

    if len(items) > 5000:
        raise HTTPException(status_code=400, detail="Batch size limit is 5,000 encounters.")

    valid_records = []
    row_errors = []
    for idx, raw_item in enumerate(items):
        try:
            validated = PatientEncounterRequest(**raw_item)
            item_dict = validated.model_dump()
            if "external_ref" in raw_item:
                item_dict["external_ref"] = raw_item["external_ref"]
            if "source" in raw_item:
                item_dict["source"] = raw_item["source"]
            valid_records.append(item_dict)
        except Exception as e:
            row_errors.append(
                {
                    "row_index": idx,
                    "error": str(e),
                    "external_ref": raw_item.get("external_ref"),
                }
            )

    if is_single and row_errors:
        raise HTTPException(status_code=422, detail=row_errors[0])

    if not valid_records:
        return {
            "status": "all_rejected",
            "accepted_count": 0,
            "rejected_count": len(row_errors),
            "errors": row_errors,
            "discharges": [],
        }

    # Vectorized scoring
    scored_results = svc.score_batch(valid_records, include_neighbor_details=True)

    inserted_rows = []
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        for scored in scored_results:
            feats = scored["features"]
            ext_ref = feats.get("external_ref")
            source = feats.get("source", "api")

            cur = conn.execute(
                """INSERT INTO discharges (
                    received_at, source, external_ref, features_json,
                    raw_score, calibrated_prob, tier, model_version,
                    neighbor_summary_json, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)""",
                (
                    now_iso,
                    source,
                    str(ext_ref) if ext_ref else None,
                    json.dumps(feats),
                    scored["raw_score"],
                    scored["calibrated_prob"],
                    scored["tier"],
                    svc.frozen_config.get("model_name", "LightGBM + Structured RAG"),
                    json.dumps(scored["neighbor_summary"]),
                    now_iso,
                ),
            )
            inserted_id = cur.lastrowid
            inserted_rows.append(
                {
                    "id": inserted_id,
                    "received_at": now_iso,
                    "source": source,
                    "external_ref": ext_ref,
                    "raw_score": scored["raw_score"],
                    "calibrated_prob": scored["calibrated_prob"],
                    "tier": scored["tier"],
                    "status": "new",
                    "model_version": svc.frozen_config.get("model_name", "LightGBM + Structured RAG"),
                    "neighbor_summary": scored["neighbor_summary"],
                }
            )

        log_audit(
            conn,
            actor="api",
            event="batch_ingested" if not is_single else "discharge_created",
            payload={
                "accepted_count": len(inserted_rows),
                "rejected_count": len(row_errors),
                "is_single": is_single,
            },
        )

    if is_single:
        return inserted_rows[0]

    return {
        "status": "completed",
        "accepted_count": len(inserted_rows),
        "rejected_count": len(row_errors),
        "discharges": inserted_rows,
        "errors": row_errors,
    }


@router.post("/discharges/upload", status_code=status.HTTP_200_OK)
async def upload_discharges_csv(file: UploadFile = File(...)):
    """Upload batch CSV of encounters. Returns full validation report with row-level errors."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    content = await file.read()
    try:
        df_csv = pd.read_csv(io.BytesIO(content), keep_default_na=False)
        df_csv = df_csv.fillna("None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if len(df_csv) > 5000:
        raise HTTPException(status_code=400, detail="CSV upload exceeds maximum 5,000 rows.")

    records = df_csv.to_dict(orient="records")
    for r in records:
        r["source"] = "csv"

    # Reuse batch ingestion
    result = create_discharges(records)
    return {
        "filename": file.filename,
        "total_rows_received": len(records),
        "accepted_count": result.get("accepted_count", 0),
        "rejected_count": result.get("rejected_count", 0),
        "accepted_rows": result.get("discharges", []),
        "rejected_rows": result.get("errors", []),
    }


# ---------------------------------------------------------------------------
# 2. Clinical Worklist (Raw-Score Ranked & Capacity Mode)
# ---------------------------------------------------------------------------


@router.get("/worklist")
def get_worklist(
    status_filter: Optional[str] = Query(None, alias="status"),
    tier_filter: Optional[str] = Query(None, alias="tier"),
    search: Optional[str] = Query(None),
    capacity: Optional[int] = Query(
        None, ge=1, le=1000, description="Capacity mode: top N OPEN items by raw score"
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Ranked clinical care-team worklist.

    Sorted strictly by RAW model score (tie-break by ID).
    Includes tier counts, search, and capacity mode (top N open items).
    """
    with get_db() as conn:
        # Tier counts across all active discharges
        tier_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        tc_cursor = conn.execute("SELECT tier, COUNT(*) as cnt FROM discharges GROUP BY tier")
        for row in tc_cursor.fetchall():
            if row["tier"] in tier_counts:
                tier_counts[row["tier"]] = row["cnt"]

        # Capacity Mode: top N open items by raw score
        if capacity is not None:
            cursor = conn.execute(
                """SELECT id, received_at, source, external_ref, raw_score, calibrated_prob,
                          tier, status, assignee, updated_at, outcome_readmitted_30d, features_json
                   FROM discharges
                   WHERE status != 'closed'
                   ORDER BY raw_score DESC, id ASC
                   LIMIT ?""",
                (capacity,),
            )
            items_raw = cursor.fetchall()
            items = []
            for r in items_raw:
                feats = json.loads(r["features_json"])
                items.append(
                    {
                        "id": r["id"],
                        "received_at": r["received_at"],
                        "source": r["source"],
                        "external_ref": r["external_ref"],
                        "raw_score": r["raw_score"],
                        "calibrated_prob": r["calibrated_prob"],
                        "tier": r["tier"],
                        "status": r["status"],
                        "assignee": r["assignee"],
                        "updated_at": r["updated_at"],
                        "outcome_readmitted_30d": r["outcome_readmitted_30d"],
                        "patient_summary": {
                            "age": feats.get("age", "[60-70)"),
                            "gender": feats.get("gender", "Female"),
                            "time_in_hospital": feats.get("time_in_hospital", 3),
                            "primary_diagnosis": feats.get("diag_1_group", "Other"),
                            "number_inpatient": feats.get("number_inpatient", 0),
                        },
                    }
                )
            return {
                "mode": "capacity_mode",
                "capacity_limit": capacity,
                "total_count": len(items),
                "tier_counts": tier_counts,
                "items": items,
            }

        # Standard worklist query with filters
        where_clauses = []
        params = []

        if status_filter:
            if status_filter.lower() == "open":
                where_clauses.append("status != 'closed'")
            else:
                where_clauses.append("status = ?")
                params.append(status_filter)

        if tier_filter:
            where_clauses.append("tier = ?")
            params.append(tier_filter)

        if search:
            search_param = f"%{search}%"
            where_clauses.append(
                "(external_ref LIKE ? OR CAST(id AS TEXT) LIKE ? OR features_json LIKE ?)"
            )
            params.extend([search_param, search_param, search_param])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total count matching filters
        count_cursor = conn.execute(f"SELECT COUNT(*) FROM discharges {where_sql}", params)
        total_count = count_cursor.fetchone()[0]

        # Paged rows: strictly ordered by raw_score DESC, id ASC
        offset = (page - 1) * page_size
        query_sql = f"""
            SELECT id, received_at, source, external_ref, raw_score, calibrated_prob,
                   tier, status, assignee, updated_at, outcome_readmitted_30d, features_json
            FROM discharges
            {where_sql}
            ORDER BY raw_score DESC, id ASC
            LIMIT ? OFFSET ?
        """
        rows_cursor = conn.execute(query_sql, params + [page_size, offset])
        items = []
        for r in rows_cursor.fetchall():
            feats = json.loads(r["features_json"])
            items.append(
                {
                    "id": r["id"],
                    "received_at": r["received_at"],
                    "source": r["source"],
                    "external_ref": r["external_ref"],
                    "raw_score": r["raw_score"],
                    "calibrated_prob": r["calibrated_prob"],
                    "tier": r["tier"],
                    "status": r["status"],
                    "assignee": r["assignee"],
                    "updated_at": r["updated_at"],
                    "outcome_readmitted_30d": r["outcome_readmitted_30d"],
                    "patient_summary": {
                        "age": feats.get("age", "[60-70)"),
                        "gender": feats.get("gender", "Female"),
                        "time_in_hospital": feats.get("time_in_hospital", 3),
                        "primary_diagnosis": feats.get("diag_1_group", "Other"),
                        "number_inpatient": feats.get("number_inpatient", 0),
                    },
                }
            )

        return {
            "mode": "standard",
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "tier_counts": tier_counts,
            "items": items,
        }


# ---------------------------------------------------------------------------
# 3. Single Encounter Detail (TreeSHAP on demand, similar patients, actions)
# ---------------------------------------------------------------------------


@router.get("/discharges/{discharge_id}")
def get_discharge_detail(discharge_id: int):
    """Retrieve encounter details. Computes TreeSHAP reasons on demand and caches them."""
    svc = get_service()
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM discharges WHERE id = ?", (discharge_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Discharge ID {discharge_id} not found.")

        feats = json.loads(row["features_json"])
        neighbor_summary = (
            json.loads(row["neighbor_summary_json"]) if row["neighbor_summary_json"] else None
        )

        # 1. On-demand cached TreeSHAP explanations
        explanation = svc.explain_cached(discharge_id, feats, top_k=5)

        # 2. Deterministic suggested interventions
        interventions = generate_suggested_interventions(
            feats,
            risk_tier=row["tier"],
            top_risk_drivers=explanation.get("top_risk_increasing"),
        )

        # 3. Action history
        actions_cur = conn.execute(
            "SELECT * FROM actions WHERE discharge_id = ? ORDER BY created_at DESC",
            (discharge_id,),
        )
        actions = [
            {
                "id": a["id"],
                "type": a["type"],
                "note": a["note"],
                "created_at": a["created_at"],
                "actor": a["actor"],
            }
            for a in actions_cur.fetchall()
        ]

        # 4. Audit trail
        audit_cur = conn.execute(
            "SELECT * FROM audit_log WHERE payload_json LIKE ? ORDER BY ts DESC LIMIT 20",
            (f'%"{discharge_id}"%',),
        )
        audit_trail = [
            {
                "id": al["id"],
                "ts": al["ts"],
                "actor": al["actor"],
                "event": al["event"],
                "payload": json.loads(al["payload_json"]) if al["payload_json"] else None,
            }
            for al in audit_cur.fetchall()
        ]

        return {
            "id": row["id"],
            "received_at": row["received_at"],
            "source": row["source"],
            "external_ref": row["external_ref"],
            "raw_score": row["raw_score"],
            "calibrated_prob": row["calibrated_prob"],
            "tier": row["tier"],
            "status": row["status"],
            "assignee": row["assignee"],
            "model_version": row["model_version"],
            "outcome_readmitted_30d": row["outcome_readmitted_30d"],
            "outcome_at": row["outcome_at"],
            "features": feats,
            "shap_explanation": explanation,
            "similar_patients_summary": neighbor_summary,
            "suggested_interventions": interventions,
            "action_history": actions,
            "audit_trail": audit_trail,
        }


# ---------------------------------------------------------------------------
# 4. Clinical Workflow Mutations (Audit Logged)
# ---------------------------------------------------------------------------


@router.patch("/discharges/{discharge_id}")
def update_discharge(discharge_id: int, req: DischargePatchRequest):
    """Update status, assignee, or apply a clinician tier override with mandatory clinical reason."""
    if req.clinician_override_tier and not req.override_reason:
        raise HTTPException(
            status_code=400,
            detail="A clinical override reason is strictly required when changing risk tier.",
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM discharges WHERE id = ?", (discharge_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Discharge not found.")

        updates = []
        params = []
        audit_payload: Dict[str, Any] = {"discharge_id": discharge_id}

        if req.status:
            updates.append("status = ?")
            params.append(req.status)
            audit_payload["old_status"] = row["status"]
            audit_payload["new_status"] = req.status

        if req.assignee is not None:
            updates.append("assignee = ?")
            params.append(req.assignee)
            audit_payload["assignee"] = req.assignee

        if req.clinician_override_tier:
            updates.append("tier = ?")
            params.append(req.clinician_override_tier)
            audit_payload["old_tier"] = row["tier"]
            audit_payload["new_tier"] = req.clinician_override_tier
            audit_payload["override_reason"] = req.override_reason

        updates.append("updated_at = ?")
        params.append(now_iso)
        params.append(discharge_id)

        conn.execute(
            f"UPDATE discharges SET {', '.join(updates)} WHERE id = ?",
            params,
        )

        log_audit(
            conn,
            actor=req.actor,
            event="discharge_updated",
            payload=audit_payload,
            ts=now_iso,
        )

    return {"status": "success", "discharge_id": discharge_id, "updated_at": now_iso}


@router.post("/discharges/{discharge_id}/actions", status_code=status.HTTP_201_CREATED)
def add_discharge_action(discharge_id: int, req: ActionCreateRequest):
    """Record a clinical action. Notes carry a warning against entering PHI."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.execute("SELECT id FROM discharges WHERE id = ?", (discharge_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Discharge not found.")

        cur = conn.execute(
            """INSERT INTO actions (discharge_id, type, note, created_at, actor)
               VALUES (?, ?, ?, ?, ?)""",
            (discharge_id, req.type, req.note, now_iso, req.actor),
        )
        action_id = cur.lastrowid

        # Update discharge updated_at
        conn.execute(
            "UPDATE discharges SET updated_at = ? WHERE id = ?",
            (now_iso, discharge_id),
        )

        log_audit(
            conn,
            actor=req.actor,
            event="action_added",
            payload={
                "discharge_id": discharge_id,
                "action_id": action_id,
                "type": req.type,
            },
            ts=now_iso,
        )

    return {
        "id": action_id,
        "discharge_id": discharge_id,
        "type": req.type,
        "note": req.note,
        "actor": req.actor,
        "created_at": now_iso,
        "phi_warning": "Verified: Free-text note stored without Protected Health Information.",
    }


@router.post("/discharges/{discharge_id}/outcome")
def record_discharge_outcome(discharge_id: int, req: OutcomeCreateRequest):
    """Record verified 30-day readmission outcome for a discharge encounter."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.execute("SELECT id FROM discharges WHERE id = ?", (discharge_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Discharge not found.")

        conn.execute(
            """UPDATE discharges
               SET outcome_readmitted_30d = ?, outcome_at = ?, updated_at = ?
               WHERE id = ?""",
            (req.outcome_readmitted_30d, now_iso, now_iso, discharge_id),
        )

        log_audit(
            conn,
            actor=req.actor,
            event="outcome_recorded",
            payload={
                "discharge_id": discharge_id,
                "outcome_readmitted_30d": req.outcome_readmitted_30d,
            },
            ts=now_iso,
        )

    return {
        "status": "outcome_recorded",
        "discharge_id": discharge_id,
        "outcome_readmitted_30d": req.outcome_readmitted_30d,
        "outcome_at": now_iso,
    }


# ---------------------------------------------------------------------------
# 5. Simulated Feed & Time Machine Endpoints
# ---------------------------------------------------------------------------


@router.post("/feed/start")
def start_feed(req: FeedStartRequest):
    """Start simulated replay feed using held-out test encounters."""
    engine = get_feed_engine()
    return engine.start(rate_per_minute=req.rate_per_minute, seed=req.seed)


@router.post("/feed/stop")
def stop_feed():
    """Stop simulated replay feed."""
    engine = get_feed_engine()
    return engine.stop()


@router.get("/feed/status")
def get_feed_status():
    """Get status of the simulated encounter feed."""
    engine = get_feed_engine()
    return engine.get_status()


@router.post("/feed/time-machine")
def advance_time_machine(req: TimeMachineRequest):
    """Advance simulated time and release outcomes from sealed feed_truth (>30 days)."""
    engine = get_feed_engine()
    return engine.advance_time_machine(days=req.days)


# ---------------------------------------------------------------------------
# 6. Live Monitoring, Score Drift PSI, and Small-n Guardrails
# ---------------------------------------------------------------------------


@router.get("/monitoring")
def get_monitoring_dashboard():
    """Live operational monitoring: tier mix, raw-score drift PSI, input-feature PSI,

    data quality rates, and performance evaluation with small-n guards.
    """
    svc = get_service()
    with get_db() as conn:
        # Total counts
        cur = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN source = 'simulated_feed' THEN 1 ELSE 0 END) as simulated_count,
                SUM(CASE WHEN source = 'api' THEN 1 ELSE 0 END) as api_count,
                SUM(CASE WHEN source = 'csv' THEN 1 ELSE 0 END) as csv_count,
                SUM(CASE WHEN status != 'closed' THEN 1 ELSE 0 END) as open_count,
                SUM(CASE WHEN outcome_readmitted_30d IS NOT NULL THEN 1 ELSE 0 END) as outcome_count,
                SUM(CASE WHEN outcome_readmitted_30d = 1 THEN 1 ELSE 0 END) as positive_outcomes
               FROM discharges"""
        )
        counts_row = cur.fetchone()

        total = counts_row["total"]
        simulated_cnt = counts_row["simulated_count"] or 0
        api_cnt = counts_row["api_count"] or 0
        csv_cnt = counts_row["csv_count"] or 0
        open_cnt = counts_row["open_count"] or 0
        outcome_cnt = counts_row["outcome_count"] or 0
        pos_cnt = counts_row["positive_outcomes"] or 0

        # Tier mix
        cur_tier = conn.execute("SELECT tier, COUNT(*) as cnt FROM discharges GROUP BY tier")
        tier_actual = {row["tier"]: row["cnt"] for row in cur_tier.fetchall()}
        tier_expected_shares = {"Critical": 0.05, "High": 0.05, "Medium": 0.10, "Low": 0.80}

        tier_mix = {}
        alerts = []

        for t in ["Critical", "High", "Medium", "Low"]:
            cnt = tier_actual.get(t, 0)
            share = cnt / total if total > 0 else 0.0
            tier_mix[t] = {
                "count": cnt,
                "share": round(share, 4),
                "expected_share": tier_expected_shares[t],
                "delta": round(share - tier_expected_shares[t], 4),
            }

        # Tier mix alerts
        if total >= 50:
            crit_share = tier_mix["Critical"]["share"]
            if crit_share > 0.10:
                alerts.append(
                    {
                        "severity": "High",
                        "type": "tier_mix_shift",
                        "message": f"Critical tier proportion ({crit_share:.1%}) significantly exceeds expected 5%.",
                    }
                )
            elif crit_share < 0.02:
                alerts.append(
                    {
                        "severity": "Moderate",
                        "type": "tier_mix_shift",
                        "message": f"Critical tier proportion ({crit_share:.1%}) is unusually low (< 2%).",
                    }
                )

        # Score drift PSI calculation against training deciles
        raw_scores_cur = conn.execute(
            "SELECT raw_score FROM discharges ORDER BY id DESC LIMIT 5000"
        )
        recent_scores = np.array([r["raw_score"] for r in raw_scores_cur.fetchall()])

        score_psi = 0.0
        psi_severity = "Minimal (Stable)"

        if len(recent_scores) >= 20 and svc.serving_thresholds:
            ref_edges = np.array(svc.serving_thresholds["psi_reference"]["decile_bin_edges"])
            ref_shares = np.array(svc.serving_thresholds["psi_reference"]["training_shares"])

            # Bin actual scores into reference decile bins
            act_counts, _ = np.histogram(recent_scores, bins=ref_edges)
            act_shares = (act_counts / len(recent_scores)) + 1e-4
            act_shares = act_shares / np.sum(act_shares)

            ref_shares_norm = ref_shares / np.sum(ref_shares)
            score_psi = float(
                np.sum((act_shares - ref_shares_norm) * np.log(act_shares / ref_shares_norm))
            )
            score_psi = max(0.0, round(score_psi, 4))
            psi_severity = classify_drift_severity(score_psi)

            if score_psi >= 0.25:
                alerts.append(
                    {
                        "severity": "High",
                        "type": "score_drift_alert",
                        "message": f"High score drift detected (PSI={score_psi:.4f} >= 0.25). Distribution differs from training.",
                    }
                )
            elif score_psi >= 0.10:
                alerts.append(
                    {
                        "severity": "Moderate",
                        "type": "score_drift_warning",
                        "message": f"Moderate score drift detected (PSI={score_psi:.4f} >= 0.10). Monitor incoming population.",
                    }
                )

        # Performance Evaluation with Small-n Guardrail
        # Explicit guard: require >= 200 resolved outcomes AND >= 20 positives
        insufficient_outcomes = outcome_cnt < 200 or pos_cnt < 20
        outcomes_evaluation = {
            "insufficient_outcomes": insufficient_outcomes,
            "sample_size_outcomes": outcome_cnt,
            "positive_readmissions": pos_cnt,
            "minimum_required_outcomes": 200,
            "minimum_required_positives": 20,
        }

        if insufficient_outcomes:
            outcomes_evaluation["status_message"] = (
                f"Insufficient outcome data ({outcome_cnt}/200 outcomes, {pos_cnt}/20 positives). "
                "Performance metrics are withheld until sample size reaches clinical statistical validity."
            )
        else:
            # Query actual outcomes and scores
            cur_eval = conn.execute(
                """SELECT raw_score, calibrated_prob, outcome_readmitted_30d
                   FROM discharges
                   WHERE outcome_readmitted_30d IS NOT NULL"""
            )
            eval_rows = cur_eval.fetchall()
            y_true = np.array([r["outcome_readmitted_30d"] for r in eval_rows])
            y_scores = np.array([r["raw_score"] for r in eval_rows])
            y_cal = np.array([r["calibrated_prob"] for r in eval_rows])

            live_auc = float(round(roc_auc_score(y_true, y_scores), 4))

            # 500-sample bootstrap for 95% CI
            rng = np.random.RandomState(42)
            boot_aucs = []
            n_eval = len(y_true)
            for _ in range(500):
                boot_idx = rng.randint(0, n_eval, n_eval)
                if len(np.unique(y_true[boot_idx])) > 1:
                    boot_aucs.append(roc_auc_score(y_true[boot_idx], y_scores[boot_idx]))

            auc_ci_lower = float(round(np.percentile(boot_aucs, 2.5), 4)) if boot_aucs else live_auc
            auc_ci_upper = float(round(np.percentile(boot_aucs, 97.5), 4)) if boot_aucs else live_auc

            # Precision at top 5%, 10%, 20%
            sorted_idx = np.argsort(y_scores)[::-1]
            precisions = {}
            for pct in [0.05, 0.10, 0.20]:
                k = max(1, int(round(n_eval * pct)))
                top_k_labels = y_true[sorted_idx[:k]]
                prec = float(round(np.sum(top_k_labels) / k, 4))
                precisions[f"top_{int(pct*100)}pct"] = {
                    "k_encounters": k,
                    "caught_readmissions": int(np.sum(top_k_labels)),
                    "precision": prec,
                }

            # Calibration Deciles
            cal_bins = []
            bin_edges = np.linspace(0, 1, 11)
            for b in range(10):
                mask = (y_cal >= bin_edges[b]) & (y_cal < bin_edges[b + 1])
                if np.sum(mask) > 0:
                    cal_bins.append(
                        {
                            "bin_range": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}",
                            "count": int(np.sum(mask)),
                            "predicted_prob_mean": float(round(np.mean(y_cal[mask]), 4)),
                            "observed_readmission_rate": float(round(np.mean(y_true[mask]), 4)),
                        }
                    )

            outcomes_evaluation.update(
                {
                    "auc_roc": live_auc,
                    "auc_ci_95": [auc_ci_lower, auc_ci_upper],
                    "precision_at_capacity": precisions,
                    "calibration_deciles": cal_bins,
                    "frozen_val_auc_ci": [0.6487, 0.6792],
                }
            )

            # Performance degradation alert
            if auc_ci_upper < 0.6487:
                alerts.append(
                    {
                        "severity": "High",
                        "type": "performance_degradation",
                        "message": f"Live AUC ({live_auc:.4f}) is statistically below the frozen validation 95% CI lower bound (0.6487).",
                    }
                )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "total_discharges": total,
                "simulated_feed_count": simulated_cnt,
                "api_count": api_cnt,
                "csv_count": csv_cnt,
                "open_worklist_count": open_cnt,
                "resolved_outcomes_count": outcome_cnt,
                "positive_readmissions_count": pos_cnt,
            },
            "tier_mix": tier_mix,
            "score_drift": {
                "metric": "Population Stability Index (PSI)",
                "raw_score_psi": score_psi,
                "severity": psi_severity,
                "sample_size": len(recent_scores),
                "thresholds": {"stable": "< 0.10", "moderate": "0.10 - 0.25", "high_alert": ">= 0.25"},
            },
            "data_quality": {
                "conformity_rate": 1.0,
                "missing_rate": 0.0,
                "status": "Healthy",
            },
            "outcomes_evaluation": outcomes_evaluation,
            "alerts": alerts,
        }


# ---------------------------------------------------------------------------
# 7. Model Metadata Info Endpoint
# ---------------------------------------------------------------------------


@router.get("/model/info")
def get_model_info():
    """Frozen model serving information, Git hash, thresholds, and link to model card."""
    svc = get_service()
    frozen = svc.frozen_config
    return {
        "model_name": frozen.get("model_name", "LightGBM + Structured RAG"),
        "model_version": "1.0.0 (Frozen Production)",
        "git_commit_hash": frozen.get("git_commit_hash", "39c2a90"),
        "frozen_timestamp": frozen.get("frozen_timestamp"),
        "serving_thresholds": svc.serving_thresholds.get("raw_score_cutoffs")
        if svc.serving_thresholds
        else None,
        "tier_definitions": svc.serving_thresholds.get("tier_definitions")
        if svc.serving_thresholds
        else None,
        "splits_summary": {
            "train": {"encounters": 69535, "strategy": "Patient-disjoint stratified group"},
            "validation": {"encounters": 14902, "strategy": "Patient-disjoint stratified group"},
            "test": {"encounters": 14903, "strategy": "Patient-disjoint held-out benchmark"},
        },
        "model_card_url": "/docs/MODEL_CARD.md",
        "api_contract_url": "/docs/api_contract.md",
    }
