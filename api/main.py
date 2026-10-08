import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    CohortPatient,
    CohortResponse,
    ExplainResponse,
    PatientEncounterRequest,
    PredictResponse,
    SimilarEncounter,
    SimilarRequest,
    SimilarResponse,
)
from api.service import get_service
from src.config import METRICS_DIR, MODELS_DIR

app = FastAPI(
    title="ReadmitIQ Decision Support API",
    description="Clinical-grade 30-day hospital readmission risk prediction with Structured RAG retrieval and SHAP explainability.",
    version="1.0.0",
)

# CORS configuration
origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000")
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """System health check and model version metadata."""
    svc = get_service()
    frozen = svc.frozen_config
    return {
        "status": "healthy",
        "version": "1.0.0",
        "git_commit_hash": frozen.get("git_commit_hash", "unknown"),
        "frozen_timestamp": frozen.get("frozen_timestamp"),
        "model_name": frozen.get("model_name"),
        "artifacts_loaded": {
            "model": svc.model is not None,
            "preprocessor": svc.preprocessor is not None,
            "calibrator": svc.calibrator is not None,
            "retriever_faiss": svc.faiss_index is not None,
            "retriever_meta": svc.train_meta is not None,
            "feature_list": svc.feature_meta is not None,
        },
    }


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict_readmission(encounter: PatientEncounterRequest):
    """Predict calibrated 30-day readmission risk, risk tier, SHAP explanations, and similar encounters."""
    svc = get_service()
    try:
        patient_dict = encounter.model_dump()
        result = svc.predict(patient_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/similar", response_model=SimilarResponse, status_code=status.HTTP_200_OK)
def find_similar_patients(req: SimilarRequest):
    """Retrieve top-k similar historical encounters from TRAIN split and calculate neighborhood readmission rate."""
    svc = get_service()
    try:
        patient_dict = req.patient_features.model_dump()
        rate, similar_list = svc.retrieve_similar(patient_dict, k=req.k)
        return {
            "k_requested": req.k,
            "neighbor_readmission_rate": round(rate, 4),
            "similar_encounters": similar_list,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.post("/explain", response_model=ExplainResponse, status_code=status.HTTP_200_OK)
def explain_patient_risk(encounter: PatientEncounterRequest, top_k: int = Query(default=3, ge=1, le=10)):
    """Generate fine-grained SHAP factor attributions for a single encounter."""
    svc = get_service()
    try:
        patient_dict = encounter.model_dump()
        explanation = svc.explain(patient_dict, top_k=top_k)
        return {
            "top_risk_increasing": explanation["top_risk_increasing"],
            "top_risk_reducing": explanation["top_risk_reducing"],
            "all_attributions": explanation["all_attributions"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


@app.get("/cohort", response_model=CohortResponse, status_code=status.HTTP_200_OK)
def retrospective_cohort_replay(
    capacity: float = Query(default=0.10, ge=0.01, le=0.50, description="Intervention team capacity fraction (1% to 50%)")
):
    """Retrospective replay on the held-out TEST split using precomputed scores.
    
    Clearly labeled as a retrospective replay of held-out historical data, not live predictions.
    """
    svc = get_service()
    if svc.cohort_df is None:
        raise HTTPException(status_code=404, detail="Cohort scored artifact (reports/demo/cohort_scored.parquet) not found.")

    df = svc.cohort_df
    total_encounters = len(df)
    n_flagged = max(1, int(total_encounters * capacity))

    flagged_df = df.head(n_flagged)
    total_positives = int(df["readmitted_30d"].sum())
    true_caught = int(flagged_df["readmitted_30d"].sum())

    precision = float(round(true_caught / n_flagged, 4))
    recall = float(round(true_caught / total_positives, 4)) if total_positives > 0 else 0.0
    prevalence = float(total_positives / total_encounters)
    lift = float(round(precision / prevalence, 2)) if prevalence > 0 else 1.0

    threshold_val = float(round(flagged_df["risk_score"].iloc[-1], 4))

    # Return top 50 patients for response payload efficiency
    patients_payload = []
    for _, row in flagged_df.head(50).iterrows():
        patients_payload.append({
            "encounter_id": int(row["encounter_id"]),
            "patient_nbr": int(row["patient_nbr"]),
            "risk_score": float(row["risk_score"]),
            "risk_tier": str(row["risk_tier"]),
            "readmitted_30d": int(row["readmitted_30d"]),
            "age": str(row["age"]),
            "gender": str(row["gender"]),
            "time_in_hospital": int(row["time_in_hospital"]),
            "number_inpatient": int(row["number_inpatient"]),
            "diag_1_group": str(row["diag_1_group"]),
        })

    return {
        "mode": "Retrospective Replay (Held-Out Test Split)",
        "capacity_fraction": capacity,
        "total_test_encounters": total_encounters,
        "flagged_encounters_count": n_flagged,
        "total_true_readmissions_in_cohort": total_positives,
        "true_readmissions_caught": true_caught,
        "precision": precision,
        "recall": recall,
        "lift_vs_random": lift,
        "threshold_at_capacity": threshold_val,
        "top_patients": patients_payload,
    }


@app.get("/metrics", status_code=status.HTTP_200_OK)
def get_model_evaluation_metrics():
    """Reads saved evaluation artifacts (ablation table, test results, fairness, drift). Zero runtime computations."""
    artifacts = {}

    ablation_p = METRICS_DIR / "ablation_table.json"
    if ablation_p.exists():
        with open(ablation_p, "r", encoding="utf-8") as f:
            artifacts["ablation_table"] = json.load(f)

    test_p = METRICS_DIR / "test_results.json"
    if test_p.exists():
        with open(test_p, "r", encoding="utf-8") as f:
            artifacts["test_results"] = json.load(f)

    fairness_p = METRICS_DIR / "fairness_audit.json"
    if fairness_p.exists():
        with open(fairness_p, "r", encoding="utf-8") as f:
            artifacts["fairness_audit"] = json.load(f)

    drift_p = METRICS_DIR / "drift_metrics.json"
    if drift_p.exists():
        with open(drift_p, "r", encoding="utf-8") as f:
            artifacts["drift_metrics"] = json.load(f)

    frozen_p = MODELS_DIR / "FROZEN_SELECTION.json"
    if frozen_p.exists():
        with open(frozen_p, "r", encoding="utf-8") as f:
            artifacts["frozen_selection"] = json.load(f)

    return artifacts
