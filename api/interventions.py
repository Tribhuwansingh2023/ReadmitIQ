"""Deterministic, rule-based decision support for hospital care transitions.

Maps clinical encounter features, risk tier, and top positive SHAP drivers
to prioritized care-transition actions. Zero generative LLMs, strictly deterministic
clinical pathways with clear trigger driver attribution.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

DISCLAIMER_TEXT = "Decision-support suggestions, not clinical advice"


def generate_suggested_interventions(
    patient_data: Dict[str, Any],
    risk_tier: str,
    top_risk_drivers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate deterministic care-transition intervention suggestions.

    Args:
        patient_data: Raw or preprocessed patient encounter attributes.
        risk_tier: Evaluated risk tier ("Critical", "High", "Medium", "Low").
        top_risk_drivers: List of top positive SHAP feature explanations.

    Returns:
        Dictionary containing disclaimer and prioritized list of suggested actions.
    """
    suggestions: List[Dict[str, Any]] = []
    seen_ids = set()

    # Extract key drivers names if available
    driver_names = set()
    if top_risk_drivers:
        for driver in top_risk_drivers:
            feature_name = driver.get("feature", "")
            if feature_name:
                driver_names.add(feature_name)

    # Helper to append unique suggestion
    def add_suggestion(
        sugg_id: str,
        action: str,
        category: str,
        priority: str,
        trigger_driver: str,
        rationale: str,
        target_timeline: str,
    ):
        if sugg_id not in seen_ids:
            seen_ids.add(sugg_id)
            suggestions.append(
                {
                    "id": sugg_id,
                    "action": action,
                    "category": category,
                    "priority": priority,
                    "trigger_driver": trigger_driver,
                    "rationale": rationale,
                    "target_timeline": target_timeline,
                }
            )

    # 1. Inpatient Utilization Rules
    num_inpatient = int(patient_data.get("number_inpatient") or 0)
    total_prior = int(patient_data.get("total_prior_visits") or 0)
    if num_inpatient >= 2 or ("number_inpatient" in driver_names and num_inpatient >= 1):
        add_suggestion(
            sugg_id="rapid_telephonic_followup",
            action="Rapid Post-Discharge Telephonic Outreach",
            category="Care Transition",
            priority="High",
            trigger_driver=f"Prior Inpatient Admissions (n={num_inpatient})",
            rationale="Patients with recurrent acute admissions exhibit higher 30-day bounce-back rates without structured 48-hour telephonic contact.",
            target_timeline="Within 48 hours post-discharge",
        )
        add_suggestion(
            sugg_id="pcp_clinic_handoff",
            action="Schedule Early Primary Care Clinic Follow-Up",
            category="Care Management",
            priority="High",
            trigger_driver=f"Frequent Acute Utilization (Total prior visits={total_prior or num_inpatient})",
            rationale="Establish confirmed outpatient medical clinic visit to prevent avoidable readmissions.",
            target_timeline="Within 7 calendar days",
        )
    elif num_inpatient == 1:
        add_suggestion(
            sugg_id="routine_telephonic_followup",
            action="Standard Post-Discharge Nurse Check-In",
            category="Care Transition",
            priority="Medium",
            trigger_driver="Prior Inpatient Stay (n=1)",
            rationale="Confirm patient understands discharge summary, medication refills, and red-flag warning symptoms.",
            target_timeline="Within 72 hours post-discharge",
        )

    # 2. Medication Titration & Glycemic Regimen Changes
    change_status = patient_data.get("change", "No")
    n_dose_changes = int(patient_data.get("n_dose_changes") or 0)
    insulin_status = patient_data.get("insulin", "No")
    if (
        change_status == "Ch"
        or n_dose_changes > 0
        or "change" in driver_names
        or "insulin" in driver_names
    ):
        insulin_note = (
            f" (Insulin status: {insulin_status})" if insulin_status in ("Up", "Down") else ""
        )
        add_suggestion(
            sugg_id="pharmacist_med_recon",
            action="Clinical Pharmacist Medication Reconciliation & Insulin Review",
            category="Pharmacotherapy",
            priority="High" if risk_tier in ("Critical", "High") else "Medium",
            trigger_driver=f"Inpatient Diabetes Regimen Change{insulin_note}",
            rationale="Active titration of anti-diabetic medications during acute stay increases outpatient administration error and hypoglycemia risk.",
            target_timeline="Prior to discharge & follow-up call at day 3",
        )

    # 3. Polypharmacy & Medication Burden
    num_medications = int(patient_data.get("num_medications") or 0)
    polypharmacy_flag = int(patient_data.get("polypharmacy") or 0)
    if num_medications >= 16 or polypharmacy_flag == 1 or "num_medications" in driver_names:
        add_suggestion(
            sugg_id="polypharmacy_review",
            action="Comprehensive Medication Regimen Simplification & Pill-Box Education",
            category="Pharmacotherapy",
            priority="Medium",
            trigger_driver=f"High Medication Burden ({num_medications} medications)",
            rationale="Complex multi-drug regimens correlate with treatment non-adherence, adverse drug reactions, and preventable readmission.",
            target_timeline="Prior to hospital discharge",
        )

    # 4. Glycemic Control & High A1C
    a1c_result = patient_data.get("A1Cresult", "None")
    max_glu = patient_data.get("max_glu_serum", "None")
    if a1c_result in (">8", ">7") or max_glu in (">300", ">200") or "A1Cresult" in driver_names:
        add_suggestion(
            sugg_id="diabetes_educator_referral",
            action="Outpatient Certified Diabetes Educator (CDCES) Referral",
            category="Endocrinology",
            priority="Medium",
            trigger_driver=f"Elevated Glycemic Marker (A1C: {a1c_result}, Glucose: {max_glu})",
            rationale="Uncontrolled glycemia benefits from outpatient self-management education, diet counseling, and consideration of CGM monitoring.",
            target_timeline="Referral placed prior to discharge",
        )

    # 5. Length of Stay / Deconditioning
    time_in_hospital = int(patient_data.get("time_in_hospital") or 0)
    if time_in_hospital >= 7 or "time_in_hospital" in driver_names:
        add_suggestion(
            sugg_id="functional_rehab_assessment",
            action="Physical Therapy / Functional Debility & Fall Risk Assessment",
            category="Rehabilitation",
            priority="Medium",
            trigger_driver=f"Prolonged Acute Hospitalization ({time_in_hospital} days)",
            rationale="Extended bed rest in diabetic patients frequently produces deconditioning, mobility deficits, and post-discharge vulnerability.",
            target_timeline="Before discharge",
        )

    # 6. Emergency Department History
    number_emergency = int(patient_data.get("number_emergency") or 0)
    if number_emergency >= 1 or "number_emergency" in driver_names:
        add_suggestion(
            sugg_id="urgent_care_action_plan",
            action="24/7 Care-Team Helpline Registration & Urgent Action Plan",
            category="Care Management",
            priority="Medium",
            trigger_driver=f"Recent Emergency Department Utilization (n={number_emergency})",
            rationale="Provide direct non-emergency triage line to intercept acute symptoms before emergency room presentation.",
            target_timeline="At discharge briefing",
        )

    # 7. Discharge Disposition (Home health, SNF, facility)
    disposition = str(patient_data.get("discharge_disposition_group", "")).lower()
    if (
        "home health" in disposition
        or "snf" in disposition
        or "rehab" in disposition
        or "discharge_disposition_group" in driver_names
    ):
        add_suggestion(
            sugg_id="home_health_warm_handoff",
            action="Home Health Agency Warm Handoff & First-Visit Confirmation",
            category="Home Care",
            priority="High",
            trigger_driver=f"Discharge Disposition ({patient_data.get('discharge_disposition_group', 'Post-acute care')})",
            rationale="Confirm agency acceptance, verify patient address, and lock in the first visiting nurse appointment within 72 hours.",
            target_timeline="Confirmed prior to patient leaving bedside",
        )

    # 8. High Peer Cohort Precedent / Structured RAG
    rag_rate = float(patient_data.get("rag_struct_rate_k20") or 0.0)
    if rag_rate >= 0.25 or "rag_struct_rate_k20" in driver_names:
        add_suggestion(
            sugg_id="peer_trajectory_case_conference",
            action="Multi-Disciplinary Care-Team Trajectory Review",
            category="Clinical Strategy",
            priority="High" if risk_tier in ("Critical", "High") else "Medium",
            trigger_driver=f"High Historical Peer Readmission Rate ({round(rag_rate * 100, 1)}% in top-20 matches)",
            rationale="Clinically similar past patients experienced elevated readmission rates; warrants review of social determinants and caregiver support.",
            target_timeline="Interdisciplinary rounds",
        )

    # 9. Fallback Baseline Action if list is short
    if len(suggestions) < 2:
        add_suggestion(
            sugg_id="standard_discharge_clarification",
            action="Teach-Back Discharge Instructions & Red-Flag Review",
            category="Patient Education",
            priority="Medium",
            trigger_driver="Standard Care-Transition Protocol",
            rationale="Ensure patient and primary caregiver verbalize red-flag signs (fever, dyspnea, acute glycemic excursions) and follow-up plan.",
            target_timeline="Day of discharge",
        )

    # Sort suggestions by priority (High first, then Medium, then Low)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s["priority"], 3))

    return {
        "disclaimer": DISCLAIMER_TEXT,
        "risk_tier": risk_tier,
        "suggestions_count": len(suggestions),
        "suggestions": suggestions,
    }


def _timeline_to_days(timeline: str) -> int:
    """Map human timeline string to approximate calendar days."""
    tl = timeline.lower()
    if "48 hours" in tl:
        return 2
    elif "72 hours" in tl:
        return 3
    elif "day 3" in tl:
        return 3
    elif "7" in tl:
        return 7
    elif "day of discharge" in tl:
        return 1
    return 3


def seed_tasks_for_discharge(
    conn: Any,
    discharge_id: int,
    patient_data: Dict[str, Any],
    risk_tier: str,
    top_risk_drivers: Optional[List[Dict[str, Any]]] = None,
    default_assignee: str = "Unassigned",
) -> List[Dict[str, Any]]:
    """Generate and persist default care-transition tasks for an encounter if none exist."""
    cur = conn.execute("SELECT COUNT(*) FROM tasks WHERE discharge_id = ?", (discharge_id,))
    if cur.fetchone()[0] > 0:
        return fetch_discharge_tasks(conn, discharge_id)

    suggested = generate_suggested_interventions(patient_data, risk_tier, top_risk_drivers)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    tasks_out = []
    for s in suggested["suggestions"]:
        days = _timeline_to_days(s.get("target_timeline", "3 days"))
        due_iso = (now + timedelta(days=days)).date().isoformat()

        cursor = conn.execute(
            """INSERT INTO tasks (
                discharge_id, title, category, priority, trigger_driver,
                rationale, due_date, owner, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (
                discharge_id,
                s["action"],
                s["category"],
                s["priority"],
                s.get("trigger_driver", ""),
                s["rationale"],
                due_iso,
                default_assignee,
                now_iso,
                now_iso,
            ),
        )
        task_id = cursor.lastrowid
        tasks_out.append(
            {
                "id": task_id,
                "discharge_id": discharge_id,
                "title": s["action"],
                "category": s["category"],
                "priority": s["priority"],
                "trigger_driver": s.get("trigger_driver", ""),
                "rationale": s["rationale"],
                "due_date": due_iso,
                "owner": default_assignee,
                "status": "pending",
                "completed_at": None,
                "completed_by": None,
                "completion_note": None,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    return tasks_out


def fetch_discharge_tasks(conn: Any, discharge_id: int) -> List[Dict[str, Any]]:
    """Retrieve all persisted tasks for a discharge encounter."""
    cur = conn.execute(
        """SELECT id, discharge_id, title, category, priority, trigger_driver,
                  rationale, due_date, owner, status, completed_at, completed_by,
                  completion_note, created_at, updated_at
           FROM tasks
           WHERE discharge_id = ?
           ORDER BY
               CASE priority WHEN 'Critical' THEN 0 WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
               id ASC""",
        (discharge_id,),
    )
    rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "discharge_id": r["discharge_id"],
            "title": r["title"],
            "category": r["category"],
            "priority": r["priority"],
            "trigger_driver": r["trigger_driver"],
            "rationale": r["rationale"],
            "due_date": r["due_date"],
            "owner": r["owner"],
            "status": r["status"],
            "completed_at": r["completed_at"],
            "completed_by": r["completed_by"],
            "completion_note": r["completion_note"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
