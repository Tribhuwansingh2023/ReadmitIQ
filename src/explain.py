import json
import sys
from pathlib import Path
from typing import Dict, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    VAL_RAG_PARQUET,
)

FEATURE_HUMAN_NAMES = {
    "number_inpatient": "prior inpatient visits",
    "number_emergency": "prior emergency visits",
    "number_outpatient": "prior outpatient visits",
    "total_prior_visits": "total prior visits",
    "time_in_hospital": "days in hospital",
    "num_medications": "medications administered",
    "num_lab_procedures": "lab procedures performed",
    "num_procedures": "clinical procedures",
    "number_diagnoses": "diagnoses recorded",
    "age_num": "patient age",
    "polypharmacy": "polypharmacy indicator",
    "n_active_drugs": "active diabetes medications",
    "n_dose_changes": "medication dose adjustments",
    "rag_struct_rate_k20": "similar patient readmission rate (RAG neighborhood)",
    "rag_readmit_rate_k20": "similar patient readmission rate (RAG neighborhood)",
    "discharge_disposition_group": "discharge destination",
    "admission_type_group": "admission type",
    "admission_source_group": "admission source",
    "diag_1_group": "primary diagnosis category",
    "diag_2_group": "secondary diagnosis category",
    "diag_3_group": "tertiary diagnosis category",
    "A1Cresult": "HbA1c test result",
    "max_glu_serum": "serum glucose test result",
    "diabetesMed": "prescribed diabetes medication",
    "change": "diabetic medication change",
    "insulin": "insulin regimen",
    "metformin": "metformin therapy",
    "race": "patient race",
    "gender": "patient gender",
    "medical_specialty": "admitting medical specialty",
}


def format_clinical_reason(feature: str, value: Union[int, float, str], shap_val: float) -> str:
    """Transform feature value and SHAP attribution into a clear, plain-language clinical explanation."""
    feat_clean = feature.replace("rag_readmit_rate_k20", "rag_struct_rate_k20")
    label = FEATURE_HUMAN_NAMES.get(feat_clean, feat_clean.replace("_", " "))
    direction = "raises" if shap_val > 0 else "lowers"

    if feat_clean == "number_inpatient":
        if int(value) == 0:
            return "No prior inpatient hospitalizations in the past year lowers risk"
        return f"{int(value)} prior inpatient admission{'s' if int(value) > 1 else ''} in the past year raises risk"
    elif feat_clean == "number_emergency":
        if int(value) == 0:
            return "No emergency room visits in past year lowers risk"
        return f"{int(value)} prior emergency visit{'s' if int(value) > 1 else ''} raises risk"
    elif feat_clean in ("rag_struct_rate_k20", "rag_readmit_rate_k20"):
        rate_pct = float(value) * 100
        if rate_pct > 15:
            return f"High readmission rate ({rate_pct:.0f}%) among top-20 similar historical patients raises risk"
        else:
            return f"Low readmission rate ({rate_pct:.0f}%) among top-20 similar historical patients lowers risk"
    elif feat_clean == "discharge_disposition_group":
        val_str = str(value).lower()
        if "home" in val_str and "health" not in val_str:
            return f"Routine discharge directly to home {direction} risk"
        return f"Discharge disposition '{value}' {direction} risk"
    elif feat_clean == "A1Cresult":
        if str(value) == "None":
            return "HbA1c was not tested during admission raises risk"
        elif ">8" in str(value):
            return "Elevated HbA1c (>8%) raises risk"
        else:
            return f"HbA1c result '{value}' {direction} risk"
    elif feat_clean == "time_in_hospital":
        return f"{int(value)} days length of stay {direction} risk"
    elif feat_clean == "num_medications":
        return f"{int(value)} distinct medications prescribed {direction} risk"
    elif feat_clean == "total_prior_visits":
        return f"{int(value)} total prior healthcare visits in past year {direction} risk"
    elif feat_clean == "diag_1_group":
        return f"Primary diagnosis of {value} {direction} risk"
    elif feat_clean == "insulin":
        if str(value) in ("Up", "Down"):
            return f"Active insulin titration ({value}) {direction} risk"
        elif str(value) == "Steady":
            return "Stable insulin regimen lowers risk"
        else:
            return f"Insulin status '{value}' {direction} risk"

    # Default fallback format
    if isinstance(value, float):
        return f"{label.capitalize()} of {value:.2f} {direction} risk"
    return f"{label.capitalize()} '{value}' {direction} risk"


class ModelExplainer:
    """SHAP explainer for the frozen final serving model."""

    def __init__(self, model_path: Optional[Path] = None, feature_list_path: Optional[Path] = None):
        if model_path is None:
            model_path = MODELS_DIR / "final" / "model.joblib"
        if feature_list_path is None:
            feature_list_path = MODELS_DIR / "final" / "feature_list.json"

        self.model = joblib.load(model_path)
        with open(feature_list_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_cols = meta["numeric_features"]
        self.cat_cols = meta["categorical_features"]
        self.all_cols = meta["all_features"]
        self.explainer = shap.TreeExplainer(self.model)

    def prepare_row(self, row: Union[Dict, pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Ensure input row matches model schema and categorical types."""
        if isinstance(row, dict):
            df = pd.DataFrame([row])
        elif isinstance(row, pd.Series):
            df = pd.DataFrame([row.to_dict()])
        else:
            df = row.copy()

        # Rename backwards-compatible column if needed
        if "rag_readmit_rate_k20" in df.columns and "rag_struct_rate_k20" not in df.columns:
            df["rag_struct_rate_k20"] = df["rag_readmit_rate_k20"]
        elif "rag_struct_rate_k20" in df.columns and "rag_readmit_rate_k20" not in df.columns:
            df["rag_readmit_rate_k20"] = df["rag_struct_rate_k20"]

        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = "Missing"
        for c in self.num_cols:
            if c not in df.columns:
                df[c] = 0.0

        df_out = df[self.all_cols].copy()
        for col in self.cat_cols:
            df_out[col] = df_out[col].astype("category")
        for col in self.num_cols:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0.0)
        return df_out

    def explain_patient(self, patient_data: Union[Dict, pd.Series], top_k: int = 3) -> Dict:
        """Compute patient-level SHAP attribution and return human-readable clinical reasons."""
        X = self.prepare_row(patient_data)
        shap_vals = self.explainer.shap_values(X)

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        impacts = []
        for col, s_val in zip(self.all_cols, sv):
            val = X[col].iloc[0]
            if isinstance(val, (np.integer, int)):
                clean_val = int(val)
            elif isinstance(val, (np.floating, float)):
                clean_val = float(val) if not pd.isna(val) else "Missing"
            else:
                clean_val = str(val) if not pd.isna(val) else "Missing"

            impacts.append(
                {
                    "feature": col,
                    "value": clean_val,
                    "shap_value": float(round(s_val, 4)),
                    "explanation": format_clinical_reason(col, val, s_val),
                }
            )

        # Separate positive (risk increasing) and negative (risk reducing)
        pos = sorted(
            [item for item in impacts if item["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True,
        )
        neg = sorted(
            [item for item in impacts if item["shap_value"] < 0], key=lambda x: x["shap_value"]
        )

        return {
            "top_risk_increasing": pos[:top_k],
            "top_risk_reducing": neg[:top_k],
            "all_attributions": {item["feature"]: item["shap_value"] for item in impacts},
        }

    def generate_global_summary_plot(
        self, val_df: pd.DataFrame, output_path: Path, max_display: int = 15
    ):
        """Generate and save global SHAP summary beeswarm plot."""
        X_val = self.prepare_row(val_df.head(1000))
        shap_vals = self.explainer.shap_values(X_val)
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif shap_vals.ndim == 3:
            sv = shap_vals[:, :, 1]
        else:
            sv = shap_vals

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv,
            X_val,
            max_display=max_display,
            show=False,
            feature_names=[FEATURE_HUMAN_NAMES.get(c, c) for c in self.all_cols],
        )
        plt.title("ReadmitIQ: Global Feature Impact (SHAP Values)", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Global SHAP summary figure saved to {output_path}")


def run_explainability_pipeline():
    """Run explainability routine, save global plot, and generate sample explanations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    explainer = ModelExplainer()
    print("Loading validation dataset for SHAP explainability audit...")
    val_df = pd.read_parquet(VAL_RAG_PARQUET)

    # 1. Global plot
    plot_path = FIGURES_DIR / "shap_summary.png"
    explainer.generate_global_summary_plot(val_df, plot_path)

    # 2. Sample explanations
    sample_patients = []
    # Pick a high risk, low risk, and medium risk patient
    for idx in [0, 5, 12, 25, 50]:
        pat = val_df.iloc[idx].to_dict()
        enc_id = int(pat.get("encounter_id", idx))
        true_label = int(pat.get("readmitted_30d", 0))
        explanation = explainer.explain_patient(pat, top_k=3)
        sample_patients.append(
            {
                "encounter_id": enc_id,
                "actual_readmitted_30d": true_label,
                "top_risk_increasing": explanation["top_risk_increasing"],
                "top_risk_reducing": explanation["top_risk_reducing"],
            }
        )

    explanations_path = METRICS_DIR / "sample_explanations.json"
    with open(explanations_path, "w", encoding="utf-8") as f:
        json.dump(sample_patients, f, indent=2)

    print(f"Sample patient explanations saved to {explanations_path}")


if __name__ == "__main__":
    run_explainability_pipeline()
