import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.schemas import (
    CohortResponse,
    ExplainResponse,
    PatientEncounterRequest,
    PredictResponse,
    SimilarRequest,
    SimilarResponse,
)
from api.service import get_service
from api.v1 import router as v1_router
from src.config import METRICS_DIR, MODELS_DIR
from src.explain import ExplainerError

logger = logging.getLogger("readmitiq.api")

app = FastAPI(
    title="ReadmitIQ Decision Support API",
    description="Clinical-grade 30-day hospital readmission risk prediction with Structured RAG retrieval and SHAP explainability.",
    version="1.0.0",
)

# Startup event to initialize SQLite database in WAL mode and seed if empty
@app.on_event("startup")
def on_startup():
    init_db()
    if os.getenv("READMITIQ_AUTO_SEED", "true").lower() in ("true", "1", "yes"):
        try:
            from scripts.seed_demo import seed_if_empty
            seed_if_empty()
        except Exception as e:
            logger.warning("Database auto-seed deferred or skipped: %s", e)

# CORS configuration
origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
)
allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_model_version_header(request, call_next):
    """Attach mandatory X-Model-Version header to all responses."""
    response = await call_next(request)
    response.headers["X-Model-Version"] = "1.0.0 (LightGBM + Structured RAG)"
    return response


# Include v1 clinical workflow router (both /v1 and /api/v1)
app.include_router(v1_router)
app.include_router(v1_router, prefix="/api")


@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/api/health", status_code=status.HTTP_200_OK, include_in_schema=False)
def health_check():
    """System health check and model version metadata."""
    svc = get_service()
    frozen = svc.frozen_config
    return {
        "status": "healthy",
        "version": "1.0.0",
        "serve_mode": svc.serve_mode,
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
def predict_readmission(
    encounter: PatientEncounterRequest,
    allow_degraded: bool = Query(default=False, description="Allow degraded mode if RAG retrieval fails"),
):
    """Predict calibrated 30-day readmission risk, risk tier, SHAP explanations, and similar encounters."""
    svc = get_service()
    try:
        patient_dict = encounter.model_dump()
        result = svc.predict(patient_dict, allow_degraded=allow_degraded)
        return result
    except (ValueError, TypeError) as ve:
        logger.warning("Inference input validation error: %s", ve, exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid encounter input data.")
    except ExplainerError as ee:
        logger.warning("Inference explainer/retrieval error: %s", ee, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Inference unavailable: RAG retrieval failed and model requires valid peer context. Degraded scoring cannot use fabricated peer evidence.",
        )
    except Exception as e:
        logger.error("Inference processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Inference processing error. Please verify encounter inputs.")


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
    except (ValueError, TypeError) as ve:
        logger.warning("Retrieval input error: %s", ve, exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid retrieval input parameters.")
    except Exception as e:
        logger.error("Retrieval processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval processing error. Please verify request inputs.")


@app.get("/explain/global", status_code=status.HTTP_200_OK)
def get_global_model_explanation():
    """Return global model explainability summary across the validation reference cohort."""
    svc = get_service()
    try:
        return svc.get_global_explanation()
    except Exception as e:
        logger.error("Global explainability error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Global explainability processing error: {e}")


@app.post("/explain", response_model=ExplainResponse, status_code=status.HTTP_200_OK)
def explain_patient_risk(
    encounter: PatientEncounterRequest,
    top_k: int = Query(default=3, ge=1, le=10),
    allow_degraded: bool = Query(default=False, description="Allow degraded mode if RAG retrieval fails"),
):
    """Generate fine-grained SHAP factor attributions for a single encounter."""
    svc = get_service()
    try:
        patient_dict = encounter.model_dump()
        explanation = svc.explain(patient_dict, top_k=top_k, allow_degraded=allow_degraded)
        return {
            "top_risk_increasing": explanation["top_risk_increasing"],
            "top_risk_reducing": explanation["top_risk_reducing"],
            "all_attributions": explanation["all_attributions"],
            "full_attribution_table": explanation.get("full_attribution_table"),
            "base_value": explanation.get("base_value"),
            "raw_margin": explanation.get("raw_margin"),
            "output_scale": explanation.get("output_scale"),
            "scale_units": explanation.get("scale_units"),
            "scale_description": explanation.get("scale_description"),
            "plain_english_summary": explanation.get("plain_english_summary"),
            "reliability": explanation.get("reliability"),
            "degraded_mode": explanation.get("degraded_mode", False),
            "degraded_reason": explanation.get("degraded_reason"),
        }
    except (ValueError, TypeError) as ve:
        logger.warning("Explainability input validation error: %s", ve, exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid encounter input data for explainability.")
    except ExplainerError as ee:
        logger.warning("Explainability explainer/retrieval error: %s", ee, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Explainability unavailable: RAG retrieval failed and model requires valid peer context. Degraded explanations cannot use fabricated peer evidence.",
        )
    except Exception as e:
        logger.error("Explainability processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Explainability processing error. Please verify encounter inputs.")


@app.get("/cohort", response_model=CohortResponse, status_code=status.HTTP_200_OK)
def retrospective_cohort_replay(
    capacity: float = Query(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Intervention team capacity fraction (1% to 50%)",
    )
):
    """Retrospective replay on the held-out TEST split using precomputed scores.
    Clearly labeled as a retrospective replay of held-out historical data, not live predictions.
    """
    svc = get_service()
    if svc.cohort_df is None:
        raise HTTPException(
            status_code=404,
            detail="Cohort scored artifact (reports/demo/cohort_scored.parquet) not found.",
        )

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
        patients_payload.append(
            {
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
            }
        )

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
            ablation_data = json.load(f)

        results = ablation_data.get("ablation_results", [])
        transformer_rows = [r for r in results if "DistilBERT" in r.get("model", "")]
        has_transformer = len(transformer_rows) > 0

        ablation_data["transformer_status"] = "completed" if has_transformer else "pending"
        ablation_data["transformer_models"] = transformer_rows
        if not has_transformer:
            ablation_data["transformer_pending_message"] = (
                "Pending Google Colab T4 GPU execution (notebooks/colab_transformer_training.ipynb). "
                "The production model was frozen before the transformer results existed."
            )

        validation_models = []
        for r in results:
            feature_str = "With RAG (k=20)" if r.get("rag_feature") else "Tabular Features (No RAG)"
            if "DistilBERT" in r.get("model", ""):
                feature_str = (
                    "Clinical Notes + RAG" if r.get("rag_feature") else "Clinical Notes Only"
                )

            ci_low = r.get("auc_ci_95", [None, None])[0]
            ci_high = r.get("auc_ci_95", [None, None])[1]

            validation_models.append(
                {
                    "model": r.get("model", ""),
                    "feature_set": feature_str,
                    "auc": r.get("auc_roc", 0.0),
                    "auc_ci_lower": ci_low,
                    "auc_ci_upper": ci_high,
                    "brier_score": r.get("brier_score"),
                    "ece": r.get("ece"),
                    "precision_at_top10": r.get("precision_top_10pct"),
                    "lift_at_top10": r.get("lift_top_10pct"),
                    "optimal_threshold": r.get("optimal_threshold"),
                    "f1_at_optimal_threshold": r.get("f1"),
                    "paired_delta_auc": r.get("paired_delta_auc"),
                    "paired_ci_95": r.get("paired_ci_95"),
                    "ci_crosses_zero": r.get("ci_crosses_zero"),
                    "train_samples": r.get("train_samples_used", 69535),
                    "val_samples": r.get("val_samples_used", 14902),
                }
            )

        ablation_data["validation_models"] = validation_models
        artifacts["ablation_table"] = ablation_data

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
