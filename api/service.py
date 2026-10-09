import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import faiss
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
)
from api.interventions import generate_suggested_interventions
from src.explain import ModelExplainer
from src.train import prepare_lgbm_data

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

        # Initialize explainer
        self.explainer = ModelExplainer(
            model_path=self.final_dir / "model.joblib",
            feature_list_path=self.final_dir / "feature_list.json",
        )

        # Preloaded cohort scored parquet for fast /cohort replay (adaptive)
        if self.is_lite and (REPORTS_DIR / "demo" / "cohort_scored_lite.parquet").exists():
            self.cohort_parquet_path = REPORTS_DIR / "demo" / "cohort_scored_lite.parquet"
        else:
            self.cohort_parquet_path = REPORTS_DIR / "demo" / "cohort_scored.parquet"

        if self.cohort_parquet_path.exists():
            self.cohort_df = pd.read_parquet(self.cohort_parquet_path)
        else:
            self.cohort_df = None

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

    def explain(self, patient_dict: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """Explain patient risk attributions, retrieving RAG neighborhood if needed."""
        if "rag_struct_rate_k20" not in patient_dict or patient_dict["rag_struct_rate_k20"] is None:
            rag_rate, _ = self.retrieve_similar(patient_dict, k=20)
            patient_dict["rag_struct_rate_k20"] = rag_rate
        return self.explainer.explain_patient(patient_dict, top_k=top_k)

    def predict(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform end-to-end calibrated prediction, RAG retrieval, and SHAP explanation."""
        # 1. Retrieve RAG neighborhood rate
        rag_rate, similar_list = self.retrieve_similar(patient_dict, k=20)
        patient_dict["rag_struct_rate_k20"] = rag_rate

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
        explanation = self.explain(patient_dict, top_k=3)

        # 5. Deterministic Decision Support Interventions
        interventions = generate_suggested_interventions(
            patient_dict,
            risk_tier=tier,
            top_risk_drivers=explanation.get("top_risk_increasing"),
        )

        # 6. Log prediction
        self.log_prediction(patient_dict, cal_prob, tier)

        return {
            "calibrated_risk_probability": round(cal_prob, 4),
            "risk_tier": tier,
            "decision_threshold": round(f1_th, 4),
            "predicted_readmission": bool(cal_prob >= f1_th),
            "capacity_thresholds": {
                "top_5pct_critical": round(th_5, 4),
                "top_10pct_high": round(th_10, 4),
                "top_20pct_medium": round(th_20, 4),
            },
            "top_risk_increasing_factors": explanation["top_risk_increasing"],
            "top_risk_reducing_factors": explanation["top_risk_reducing"],
            "similar_patients_summary": {
                "k_neighbors": 20,
                "neighbor_readmission_rate": round(rag_rate, 4),
                "top_neighbors": similar_list[:5],
            },
            "suggested_interventions": interventions,
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
