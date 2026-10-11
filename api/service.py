import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import faiss
import joblib
import numpy as np
import pandas as pd

from api.interventions import generate_suggested_interventions
from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
)
from src.explain import ExplainerError, ModelExplainer, check_explanation_reliability
from src.train import prepare_lgbm_data

logger = logging.getLogger("readmitiq.service")

LOGS_DIR = Path(os.getenv("READMITIQ_LOG_DIR", str(Path(__file__).resolve().parent / "logs")))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG_PATH = LOGS_DIR / "predictions.jsonl"


class ModelService:
    """Core ML service managing frozen serving models, FAISS retriever, explainability, and logging."""

    def __init__(self):
        self.final_dir = MODELS_DIR / "final"
        self.frozen_meta_path = MODELS_DIR / "FROZEN_SELECTION.json"

        # Serving mode: "full" (default) or "lite" for low-memory environments
        self.serve_mode = os.getenv("READMITIQ_SERVE_MODE", "full").lower().strip()
        self.is_lite = self.serve_mode in ["lite", "l", "compact"]

        # Load frozen configuration
        with open(self.frozen_meta_path, "r", encoding="utf-8") as f:
            self.frozen_config = json.load(f)

        # Configure FAISS thread count for fast parallel batch retrieval
        try:
            num_threads = min(os.cpu_count() or 4, 8)
            faiss.omp_set_num_threads(num_threads)
        except Exception:
            pass

        # Load models and preprocessors (always lightweight, < 350 KB total)
        self.model = joblib.load(self.final_dir / "model.joblib")
        self.preprocessor = joblib.load(self.final_dir / "preprocessor.joblib")
        self.base_preprocessor = joblib.load(self.final_dir / "preprocessor_base.joblib")
        self.calibrator = joblib.load(self.final_dir / "calibrator.joblib")
        self.svd = joblib.load(self.final_dir / "structured_svd.joblib")

        # Load FAISS index & metadata (adaptive to full vs lite mode)
        if self.is_lite and (self.final_dir / "retriever_faiss_lite.index").exists():
            self.faiss_index = faiss.read_index(str(self.final_dir / "retriever_faiss_lite.index"))
            self.train_meta = pd.read_parquet(self.final_dir / "retriever_meta_lite.parquet")
        else:
            self.faiss_index = faiss.read_index(str(self.final_dir / "retriever_faiss.index"))
            self.train_meta = pd.read_parquet(self.final_dir / "retriever_meta.parquet")

        with open(self.final_dir / "feature_list.json", "r", encoding="utf-8") as f:
            self.feature_meta = json.load(f)

        self.num_cols = self.feature_meta["numeric_features"]
        self.cat_cols = self.feature_meta["categorical_features"]
        self.all_cols = self.feature_meta["all_features"]

        # Base features required by SVD preprocessor (without RAG feature)
        self.base_num_cols = [c for c in self.num_cols if c != "rag_struct_rate_k20"]

        # Load serving raw-score thresholds
        serving_th_path = self.final_dir / "serving_thresholds.json"
        if serving_th_path.exists():
            with open(serving_th_path, "r", encoding="utf-8") as f:
                self.serving_thresholds = json.load(f)
        else:
            self.serving_thresholds = None

        # Initialize explainer
        self.explainer = ModelExplainer(
            model_path=self.final_dir / "model.joblib",
            feature_list_path=self.final_dir / "feature_list.json",
        )
        self._shap_cache: Dict[Any, Dict[str, Any]] = {}

        # Preloaded cohort scored parquet for fast /cohort replay (adaptive)
        if self.is_lite and (REPORTS_DIR / "demo" / "cohort_scored_lite.parquet").exists():
            self.cohort_parquet_path = REPORTS_DIR / "demo" / "cohort_scored_lite.parquet"
        else:
            self.cohort_parquet_path = REPORTS_DIR / "demo" / "cohort_scored.parquet"

        if self.cohort_parquet_path.exists():
            self.cohort_df = pd.read_parquet(self.cohort_parquet_path)
        else:
            self.cohort_df = None

    def assign_tier(self, raw_score: float) -> str:
        """Assign risk tier based on frozen validation raw-score cutoffs (Top 5%, 10%, 20%)."""
        if self.serving_thresholds and "raw_score_cutoffs" in self.serving_thresholds:
            cutoffs = self.serving_thresholds["raw_score_cutoffs"]
            if raw_score >= cutoffs["top_5pct"]:
                return "Critical"
            elif raw_score >= cutoffs["top_10pct"]:
                return "High"
            elif raw_score >= cutoffs["top_20pct"]:
                return "Medium"
            return "Low"
        # Fallbacks matching serving_thresholds.json
        if raw_score >= 0.683933:
            return "Critical"
        elif raw_score >= 0.640775:
            return "High"
        elif raw_score >= 0.579340:
            return "Medium"
        return "Low"

    def score_batch(
        self, encounters: List[Dict[str, Any]], include_neighbor_details: bool = False
    ) -> List[Dict[str, Any]]:
        """Vectorized live scoring through SVD-FAISS retriever, LightGBM, and Isotonic calibration.

        High-throughput: processes 1,000 rows in ~100-300 ms (< 2.0s target).
        """
        if isinstance(encounters, pd.DataFrame):
            df = encounters.copy()
        else:
            df = pd.DataFrame(encounters)

        # Batch ensure required columns exist
        missing_cats = [c for c in self.cat_cols if c not in df.columns]
        for c in missing_cats:
            df[c] = "Missing"

        missing_nums = [c for c in self.base_num_cols if c not in df.columns]
        for c in missing_nums:
            df[c] = 0.0

        df[self.cat_cols] = df[self.cat_cols].fillna("Missing").astype(str)
        df[self.base_num_cols] = df[self.base_num_cols].fillna(0.0).astype(float)

        # 1. Vectorized FAISS retrieval
        missing_rag = "rag_struct_rate_k20" not in df.columns or df["rag_struct_rate_k20"].isna().any()
        neighbor_indices = None
        neighbor_sims = None

        if missing_rag:
            try:
                dense_feats = self.base_preprocessor.transform(df[self.base_num_cols + self.cat_cols])
                if hasattr(dense_feats, "toarray"):
                    dense_feats = dense_feats.toarray()
                query_vecs = self.svd.transform(dense_feats).astype(np.float32)
                faiss.normalize_L2(query_vecs)
                scores, indices = self.faiss_index.search(query_vecs, 20)
                neighbor_indices = indices
                neighbor_sims = scores
                # Vectorized neighbor readmission rate across top 20 train neighbors
                rag_rates = self.train_meta["readmitted_30d"].values[indices].mean(axis=1)
                df["rag_struct_rate_k20"] = rag_rates
            except Exception as e:
                logger.error("Vectorized FAISS retrieval failed: %s", e, exc_info=True)
                raise RuntimeError(
                    f"RAG retrieval failed for batch scoring: {e}. Model requires valid peer context."
                ) from e
        else:
            if df["rag_struct_rate_k20"].isna().any():
                raise ValueError("Batch scoring contains invalid or missing rag_struct_rate_k20 values.")
            df["rag_struct_rate_k20"] = pd.to_numeric(df["rag_struct_rate_k20"], errors="raise").astype(float)

        # 2. Vectorized LightGBM feature prep & inference
        X_lgbm = prepare_lgbm_data(df, self.num_cols, self.cat_cols)
        raw_scores = self.model.predict_proba(X_lgbm)[:, 1]
        cal_probs = self.calibrator.transform(raw_scores)

        # 3. Vectorized tier assignment
        cutoffs = (
            self.serving_thresholds["raw_score_cutoffs"]
            if self.serving_thresholds and "raw_score_cutoffs" in self.serving_thresholds
            else {"top_5pct": 0.683933, "top_10pct": 0.640775, "top_20pct": 0.579340}
        )
        th_5, th_10, th_20 = cutoffs["top_5pct"], cutoffs["top_10pct"], cutoffs["top_20pct"]
        tiers = np.where(
            raw_scores >= th_5,
            "Critical",
            np.where(raw_scores >= th_10, "High", np.where(raw_scores >= th_20, "Medium", "Low")),
        )

        rag_vals = df["rag_struct_rate_k20"].values

        # If neighbor details requested (for single patient view)
        meta_dict = None
        if include_neighbor_details and neighbor_indices is not None:
            if not hasattr(self, "_train_meta_records"):
                self._train_meta_records = self.train_meta.to_dict(orient="records")
            meta_dict = self._train_meta_records

        results = []
        for i in range(len(df)):
            raw_s = float(raw_scores[i])
            cal_p = float(cal_probs[i])
            tier = str(tiers[i])
            r_rate = float(rag_vals[i])

            neighbor_summary = {
                "k_neighbors": 20,
                "neighbor_readmission_rate": round(r_rate, 4),
            }

            if meta_dict is not None and neighbor_indices is not None:
                top_idx = neighbor_indices[i][:5]
                top_scores = neighbor_sims[i][:5]
                top_list = []
                for j, idx_val in enumerate(top_idx):
                    row = meta_dict[idx_val]
                    top_list.append(
                        {
                            "encounter_id": int(row.get("encounter_id", 0)),
                            "similarity_score": round(float(top_scores[j]), 3),
                            "age": str(row.get("age", "[60-70)")),
                            "gender": str(row.get("gender", "Female")),
                            "time_in_hospital": int(row.get("time_in_hospital", 3)),
                            "primary_diagnosis": str(row.get("diag_1_group", "Other")),
                            "prior_inpatient": int(row.get("number_inpatient", 0)),
                            "readmitted_30d": int(row.get("readmitted_30d", 0)),
                        }
                    )
                neighbor_summary["top_neighbors"] = top_list

            results.append(
                {
                    "raw_score": round(raw_s, 6),
                    "calibrated_prob": round(cal_p, 4),
                    "tier": tier,
                    "rag_struct_rate_k20": round(r_rate, 4),
                    "neighbor_summary": neighbor_summary,
                    "features": encounters[i],
                }
            )

        return results

    def explain_cached(
        self,
        cache_key: Any,
        patient_dict: Dict[str, Any],
        top_k: int = 3,
        allow_degraded: bool = False,
        force_degraded: bool = False,
        calibrated_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute TreeSHAP explanations on demand and cache them."""
        if cache_key in self._shap_cache:
            return self._shap_cache[cache_key]

        explanation = self.explain(
            patient_dict,
            top_k=top_k,
            allow_degraded=allow_degraded,
            force_degraded=force_degraded,
            calibrated_prob=calibrated_prob,
        )
        self._shap_cache[cache_key] = explanation
        return explanation

    def retrieve_similar(
        self, patient_dict: Dict[str, Any], k: int = 20
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Query FAISS index for k most similar historical train encounters using 64-dim SVD cosine similarity."""
        # Build base 141-dim features
        df_single = pd.DataFrame([patient_dict])
        for c in self.cat_cols:
            if c not in df_single.columns:
                df_single[c] = "Missing"
        for c in self.base_num_cols:
            if c not in df_single.columns:
                df_single[c] = 0.0

        # Transform using fitted base preprocessor (41 features -> 141 dims) and SVD (141 dims -> 64 dims)
        dense_feats = self.base_preprocessor.transform(
            df_single[self.base_num_cols + self.cat_cols]
        )
        if hasattr(dense_feats, "toarray"):
            dense_feats = dense_feats.toarray()

        query_vec = self.svd.transform(dense_feats).astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Search index
        scores, indices = self.faiss_index.search(query_vec, k)
        match_idx = indices[0]
        sim_scores = scores[0]

        neighbors_meta = self.train_meta.iloc[match_idx].copy()
        neighbors_meta["similarity"] = sim_scores

        readmit_rate = float(np.mean(neighbors_meta["readmitted_30d"]))

        similar_list = []
        for _, row in neighbors_meta.iterrows():
            similar_list.append(
                {
                    "encounter_id": int(row.get("encounter_id", 0)),
                    "similarity_score": round(float(row.get("similarity", 0.0)), 3),
                    "age": str(row.get("age", "[60-70)")),
                    "gender": str(row.get("gender", "Female")),
                    "time_in_hospital": int(row.get("time_in_hospital", 3)),
                    "primary_diagnosis": str(row.get("diag_1_group", "Other")),
                    "prior_inpatient": int(row.get("number_inpatient", 0)),
                    "readmitted_30d": int(row.get("readmitted_30d", 0)),
                }
            )

        return readmit_rate, similar_list

    def explain(
        self,
        patient_dict: Dict[str, Any],
        top_k: int = 3,
        allow_degraded: bool = False,
        force_degraded: bool = False,
        calibrated_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Explain patient risk attributions, retrieving RAG neighborhood if needed.

        The frozen serving model requires valid peer context (rag_struct_rate_k20).
        If RAG retrieval fails and no precomputed rag_struct_rate_k20 is provided:
        - Raises ExplainerError to fail safely.
        - Strictly prohibits fabrications (such as 0.0 or population mean) which corrupt model attributions.
        """
        if not isinstance(patient_dict, dict):
            raise ExplainerError("patient_dict must be a dictionary")
        patient_copy = dict(patient_dict)

        if "rag_struct_rate_k20" not in patient_copy or patient_copy["rag_struct_rate_k20"] is None:
            try:
                rag_rate, _ = self.retrieve_similar(patient_copy, k=20)
                patient_copy["rag_struct_rate_k20"] = rag_rate
            except Exception as e:
                logger.error("RAG retrieval failed during explain: %s", e, exc_info=True)
                raise ExplainerError(
                    f"RAG neighborhood retrieval failed: {e}. "
                    "Explainability requires valid peer context (rag_struct_rate_k20) "
                    "and cannot safely generate attributions using fabricated peer evidence."
                ) from e

        explanation = self.explainer.explain_patient(
            patient_copy, top_k=top_k, calibrated_prob=calibrated_prob
        )
        explanation["degraded_mode"] = False
        explanation["degraded_reason"] = None
        explanation["reliability"] = check_explanation_reliability(
            patient_copy,
            has_prediction=True,
            has_shap=True,
            has_rag=True,
            calibrated_prob=calibrated_prob,
            degraded_mode=False,
        )
        return explanation

    def get_global_explanation(self) -> Dict[str, Any]:
        """Return reference cohort global feature importances, distributions, and model metadata."""
        return self.explainer.get_global_explanation()

    def predict(
        self, patient_dict: Dict[str, Any], allow_degraded: bool = False
    ) -> Dict[str, Any]:
        """Perform end-to-end calibrated prediction, RAG retrieval, and SHAP explanation.

        The frozen serving model requires valid peer context (rag_struct_rate_k20).
        Because no separately validated non-RAG serving model exists in models/final,
        if RAG retrieval fails and no valid precomputed rag_struct_rate_k20 is provided:
        - The prediction fails safely by raising ExplainerError.
        - Never fabricates a feature value (such as 0.0 or 0.1139) to prevent ungrounded clinical inference.
        """
        rag_rate = patient_dict.get("rag_struct_rate_k20")
        similar_list = []

        if rag_rate is None:
            try:
                rag_rate, similar_list = self.retrieve_similar(patient_dict, k=20)
                patient_dict["rag_struct_rate_k20"] = rag_rate
            except Exception as e:
                logger.error("RAG retrieval failed during predict: %s", e, exc_info=True)
                raise ExplainerError(
                    f"RAG neighborhood retrieval failed: {e}. "
                    "The frozen serving model requires valid peer context (rag_struct_rate_k20) "
                    "and no separately validated non-RAG model exists. "
                    "Prediction fails safely to prevent clinical scoring on fabricated peer evidence."
                ) from e
        else:
            try:
                _, similar_list = self.retrieve_similar(patient_dict, k=20)
            except Exception as e:
                logger.warning("Neighbor detail retrieval failed: %s", e)
                similar_list = []

        # 2. Format for LightGBM
        df_single = pd.DataFrame([patient_dict])
        for c in self.cat_cols:
            if c not in df_single.columns:
                df_single[c] = "Missing"
        for c in self.num_cols:
            if c not in df_single.columns:
                df_single[c] = 0.0

        X_lgbm = prepare_lgbm_data(df_single, self.num_cols, self.cat_cols)

        raw_prob = float(self.model.predict_proba(X_lgbm)[0, 1])
        cal_prob = float(self.calibrator.transform(np.array([raw_prob]))[0])

        # 3. Determine Risk Tier based on Frozen Capacity Thresholds
        th_5 = self.frozen_config["thresholds"]["capacity_top_5pct_threshold"]
        th_10 = self.frozen_config["thresholds"]["capacity_top_10pct_threshold"]
        th_20 = self.frozen_config["thresholds"]["capacity_top_20pct_threshold"]
        f1_th = self.frozen_config["thresholds"]["optimal_f1_decision_threshold"]

        if cal_prob >= th_5:
            tier = "Critical"
        elif cal_prob >= th_10:
            tier = "High"
        elif cal_prob >= th_20:
            tier = "Medium"
        else:
            tier = "Low"

        # 4. SHAP Explanation
        explanation = self.explain(patient_dict, top_k=5, calibrated_prob=cal_prob)

        # 5. Deterministic Decision Support Interventions
        interventions = generate_suggested_interventions(
            patient_dict,
            risk_tier=tier,
            top_risk_drivers=explanation.get("top_risk_increasing"),
        )

        # 6. Log prediction
        self.log_prediction(patient_dict, cal_prob, tier)

        similar_summary: Dict[str, Any] = {
            "k_neighbors": 20,
            "neighbor_readmission_rate": round(float(rag_rate), 4),
            "top_neighbors": similar_list[:5],
        }

        top_increasing = explanation["top_risk_increasing"]
        top_reducing = explanation["top_risk_reducing"]

        return {
            "raw_score": round(raw_prob, 6),
            "raw_lgbm_score": round(raw_prob, 6),
            "raw_margin": explanation.get("raw_margin", round(raw_prob, 6)),
            "base_value": explanation.get("base_value", -0.1960),
            "output_scale": explanation.get("output_scale", "log_odds_margin"),
            "scale_units": explanation.get("scale_units", "Log-Odds Margin Space"),
            "scale_description": explanation.get("scale_description"),
            "plain_english_summary": explanation.get("plain_english_summary"),
            "full_attribution_table": explanation.get("full_attribution_table", []),
            "explanation_reliability": explanation.get("reliability"),
            "calibrated_risk_probability": round(cal_prob, 4),
            "risk_tier": tier,
            "decision_threshold": round(f1_th, 4),
            "predicted_readmission": bool(cal_prob >= f1_th),
            "capacity_thresholds": {
                "top_5pct_critical": round(th_5, 4),
                "top_10pct_high": round(th_10, 4),
                "top_20pct_medium": round(th_20, 4),
            },
            "top_risk_increasing_factors": top_increasing,
            "top_risk_reducing_factors": top_reducing,
            "all_attributions": explanation.get("all_attributions", {}),
            "similar_patients_summary": similar_summary,
            "suggested_interventions": interventions,
            "degraded_mode": False,
            "degraded_reason": None,
        }

    def log_prediction(self, patient_dict: Dict[str, Any], prob: float, tier: str):
        """Append anonymized inference record to JSONL log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibrated_risk_probability": round(prob, 4),
            "risk_tier": tier,
            "features_logged": {
                k: patient_dict[k]
                for k in [
                    "time_in_hospital",
                    "number_inpatient",
                    "diag_1_group",
                    "A1Cresult",
                    "rag_struct_rate_k20",
                ]
                if k in patient_dict
            },
        }
        try:
            with open(PRED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass


_SERVICE_INSTANCE = None


def get_service() -> ModelService:
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = ModelService()
    return _SERVICE_INSTANCE
