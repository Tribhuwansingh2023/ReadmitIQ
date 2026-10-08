import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    AGE_MIDPOINTS,
    ALL_DRUG_COLS,
    COUNT_COLS_TO_CAP,
    DROP_COLS,
    EXCLUDE_DISCHARGE_IDS,
)


def map_icd9_to_category(code: Optional[str]) -> str:
    """Map raw ICD-9 diagnosis string to clinical group."""
    if pd.isna(code) or code is None:
        return "other"
    code_str = str(code).strip()
    if not code_str or code_str == "?":
        return "other"

    if code_str.startswith("250"):
        return "diabetes"
    if code_str.startswith(("V", "E")):
        return "other"

    try:
        num = float(code_str.split(".")[0])
    except ValueError:
        return "other"

    if (390 <= num <= 459) or num == 785:
        return "circulatory"
    if (460 <= num <= 519) or num == 786:
        return "respiratory"
    if (520 <= num <= 579) or num == 787:
        return "digestive"
    if (580 <= num <= 629) or num == 788:
        return "genitourinary"
    if 140 <= num <= 239:
        return "neoplasms"
    if 710 <= num <= 739:
        return "musculoskeletal"
    if 800 <= num <= 999:
        return "injury"

    return "other"


def map_age_to_midpoint(age_str: str) -> float:
    """Convert age interval string to midpoint numeric value."""
    return float(AGE_MIDPOINTS.get(str(age_str).strip(), 55.0))


def group_admission_type(val: int) -> str:
    """Group admission type IDs into clinical categories."""
    try:
        val_int = int(val)
    except (ValueError, TypeError):
        return "other_unknown"
    if val_int in [1, 2, 7]:
        return "emergency_urgent"
    if val_int == 3:
        return "elective"
    if val_int == 4:
        return "newborn"
    return "other_unknown"


def group_discharge_disposition(val: int) -> str:
    """Group discharge disposition IDs into discharge status categories."""
    try:
        val_int = int(val)
    except (ValueError, TypeError):
        return "other_unknown"
    if val_int == 1:
        return "home"
    if val_int in [6, 8]:
        return "home_health"
    if val_int in [2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30]:
        return "transferred_facility"
    if val_int == 7:
        return "left_ama"
    return "other_unknown"


def group_admission_source(val: int) -> str:
    """Group admission source IDs into origin categories."""
    try:
        val_int = int(val)
    except (ValueError, TypeError):
        return "other_unknown"
    if val_int == 7:
        return "emergency_room"
    if val_int in [1, 2, 3]:
        return "referral"
    if val_int in [4, 5, 6, 10, 22, 25]:
        return "transfer"
    return "other_unknown"


def engineer_features(
    df: pd.DataFrame,
    capping_thresholds: Optional[Dict[str, float]] = None,
    is_train: bool = False,
    retained_drugs: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """Clean and transform raw diabetic data features without label leakage.

    Returns the transformed dataframe, 99th percentile capping thresholds,
    and the list of retained drug columns.
    """
    data = df.copy()

    # Drop explicit hospice/expired and invalid gender if present
    if "discharge_disposition_id" in data.columns:
        data = data[~data["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)]
    if "gender" in data.columns:
        data = data[data["gender"] != "Unknown/Invalid"]

    # Target definition: readmitted <30 days
    if "readmitted" in data.columns:
        data["readmitted_30d"] = (data["readmitted"] == "<30").astype(int)

    # Calculate active medications and dose changes before dropping any drug columns
    present_drugs = [d for d in ALL_DRUG_COLS if d in data.columns]
    data["n_active_drugs"] = (
        data[present_drugs].isin(["Up", "Down", "Steady"]).sum(axis=1).astype(int)
    )
    data["n_dose_changes"] = (
        data[present_drugs].isin(["Up", "Down"]).sum(axis=1).astype(int)
    )

    # Determine retained drugs (drugs with <= 99.9% 'No')
    if is_train or retained_drugs is None:
        retained = []
        for d in present_drugs:
            if d not in ["examide", "citoglipton"]:
                no_ratio = (data[d] == "No").mean()
                if no_ratio <= 0.999:
                    retained.append(d)
        retained_drugs = retained

    # Drop sparse or constant drug columns
    drugs_to_drop = [d for d in present_drugs if d not in retained_drugs]
    cols_to_drop = [c for c in DROP_COLS + drugs_to_drop if c in data.columns]
    data = data.drop(columns=cols_to_drop)

    # Categorical groupings
    data["age_num"] = data["age"].apply(map_age_to_midpoint)
    data["admission_type_group"] = data["admission_type_id"].apply(group_admission_type)
    data["discharge_disposition_group"] = data["discharge_disposition_id"].apply(
        group_discharge_disposition
    )
    data["admission_source_group"] = data["admission_source_id"].apply(
        group_admission_source
    )

    # ICD-9 diagnosis grouping
    for diag in ["diag_1", "diag_2", "diag_3"]:
        data[f"{diag}_group"] = data[diag].apply(map_icd9_to_category)

    # Missing value handling
    data["race"] = data["race"].fillna("Missing")
    data["medical_specialty"] = data["medical_specialty"].fillna("Missing")
    data["payer_code"] = data["payer_code"].fillna("Missing")
    data["max_glu_serum"] = data["max_glu_serum"].fillna("None")
    data["A1Cresult"] = data["A1Cresult"].fillna("None")

    # Ratio and prior visits engineering
    data["total_prior_visits"] = (
        data["number_outpatient"] + data["number_emergency"] + data["number_inpatient"]
    )
    time_hosp = data["time_in_hospital"].replace(0, 1)
    data["meds_per_day"] = data["num_medications"] / time_hosp
    data["labs_per_day"] = data["num_lab_procedures"] / time_hosp
    data["polypharmacy"] = (data["num_medications"] >= 10).astype(int)
    data["a1c_tested"] = (data["A1Cresult"] != "None").astype(int)

    # 99th percentile capping (computed strictly on train set)
    if is_train or capping_thresholds is None:
        capping_thresholds = {}
        for col in COUNT_COLS_TO_CAP:
            if col in data.columns:
                capping_thresholds[col] = float(np.percentile(data[col], 99))

    for col, cap_val in capping_thresholds.items():
        if col in data.columns:
            data[col] = np.clip(data[col], a_min=None, a_max=cap_val)

    # Drop intermediate raw ID/diag/age columns now represented by groupings
    raw_to_drop = [
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "diag_1",
        "diag_2",
        "diag_3",
    ]
    data = data.drop(columns=[c for c in raw_to_drop if c in data.columns])

    return data, capping_thresholds, retained_drugs


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separate model feature columns into numeric and categorical lists.

    Excludes identifier columns (encounter_id, patient_nbr) and target.
    """
    non_features = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "readmitted_30d",
        "fold",
    }
    candidates = [c for c in df.columns if c not in non_features]

    numeric_cols = []
    categorical_cols = []
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return sorted(numeric_cols), sorted(categorical_cols)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapse rare levels in categorical features before one-hot encoding.

    Fit strictly on TRAIN data:
    - medical_specialty: keep top 10 most frequent non-missing levels + "Other" + "Missing".
    - payer_code: group levels with < 1% train frequency into "Other".
    """

    def __init__(
        self,
        cat_cols: Optional[List[str]] = None,
        top_n_specialties: int = 10,
        payer_min_freq: float = 0.01,
    ):
        self.cat_cols = cat_cols
        self.top_n_specialties = top_n_specialties
        self.payer_min_freq = payer_min_freq
        self.top_specialties_: List[str] = []
        self.frequent_payers_: List[str] = []

    def fit(self, X, y=None):
        is_df = isinstance(X, pd.DataFrame)
        cols = list(X.columns) if is_df else (self.cat_cols or [f"col_{i}" for i in range(X.shape[1])])
        X_df = X if is_df else pd.DataFrame(X, columns=cols)

        if "medical_specialty" in cols:
            vc = X_df["medical_specialty"].value_counts()
            non_missing = vc[~vc.index.isin(["Missing", "Other"])]
            self.top_specialties_ = non_missing.nlargest(self.top_n_specialties).index.tolist()

        if "payer_code" in cols:
            n_rows = len(X_df)
            pvc = X_df["payer_code"].value_counts()
            self.frequent_payers_ = pvc[(pvc / n_rows) >= self.payer_min_freq].index.tolist()

        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        cols = list(X.columns) if is_df else (self.cat_cols or [f"col_{i}" for i in range(X.shape[1])])
        X_df = X.copy() if is_df else pd.DataFrame(X, columns=cols)

        if "medical_specialty" in cols and self.top_specialties_:
            allowed = set(self.top_specialties_) | {"Missing", "Other"}
            X_df["medical_specialty"] = X_df["medical_specialty"].apply(
                lambda v: v if v in allowed else "Other"
            )

        if "payer_code" in cols and self.frequent_payers_:
            allowed = set(self.frequent_payers_) | {"Other"}
            X_df["payer_code"] = X_df["payer_code"].apply(
                lambda v: v if v in allowed else "Other"
            )

        return X_df if is_df else X_df.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.cat_cols is not None:
            return np.asarray(self.cat_cols, dtype=object)
        return np.asarray([], dtype=object)


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric scaling and categorical encoding with rare level collapsing."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("grouper", RareCategoryGrouper(cat_cols=categorical_cols)),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


# Alias for pipeline builder
build_preprocessor_pipeline = build_preprocessor


def build_base_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessor for base clinical features (without RAG)."""
    num_cols, cat_cols = get_feature_columns(df)
    num_cols = [c for c in num_cols if not c.startswith("rag_")]
    return build_preprocessor(num_cols, cat_cols), num_cols, cat_cols


def build_rag_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessor including engineered RAG features."""
    num_cols, cat_cols = get_feature_columns(df)
    return build_preprocessor(num_cols, cat_cols), num_cols, cat_cols
