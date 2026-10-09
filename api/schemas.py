from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator


class PatientEncounterRequest(BaseModel):
    """Raw clinical encounter features for real-time inference."""

    age: str = Field(default="[60-70)")
    gender: str = Field(default="Female")
    race: str = Field(default="Caucasian")
    time_in_hospital: int = Field(default=3, ge=1, le=14)
    num_lab_procedures: int = Field(default=40, ge=1, le=150)
    num_procedures: int = Field(default=1, ge=0, le=10)
    num_medications: int = Field(default=12, ge=1, le=100)
    number_outpatient: int = Field(default=0, ge=0)
    number_emergency: int = Field(default=0, ge=0)
    number_inpatient: int = Field(default=0, ge=0)
    number_diagnoses: int = Field(default=7, ge=1, le=20)
    admission_type_group: str = Field(default="Emergency")
    admission_source_group: str = Field(default="Emergency Room")
    discharge_disposition_group: str = Field(default="Home")
    medical_specialty: str = Field(default="InternalMedicine")
    payer_code: str = Field(default="Medicare")
    diag_1_group: str = Field(default="Circulatory")
    diag_2_group: str = Field(default="Diabetes")
    diag_3_group: str = Field(default="Other")
    max_glu_serum: str = Field(default="None")
    A1Cresult: str = Field(default="None")
    diabetesMed: str = Field(default="Yes")
    change: str = Field(default="No")
    insulin: str = Field(default="No")
    metformin: str = Field(default="No")
    glipizide: str = Field(default="No")
    glyburide: str = Field(default="No")
    glimepiride: str = Field(default="No")
    pioglitazone: str = Field(default="No")
    rosiglitazone: str = Field(default="No")
    acarbose: str = Field(default="No")
    repaglinide: str = Field(default="No")
    nateglinide: str = Field(default="No")
    glyburide_metformin: str = Field(default="No", alias="glyburide-metformin")

    # Optional pre-computed derived features
    age_num: Optional[float] = None
    labs_per_day: Optional[float] = None
    meds_per_day: Optional[float] = None
    total_prior_visits: Optional[int] = None
    polypharmacy: Optional[int] = None
    n_active_drugs: Optional[int] = None
    n_dose_changes: Optional[int] = None
    a1c_tested: Optional[int] = None
    rag_struct_rate_k20: Optional[float] = None

    class Config:
        populate_by_name = True

    @model_validator(mode="after")
    def compute_derived_fields(self):
        # Age numeric midpoint
        if self.age_num is None:
            age_map = {
                "[0-10)": 5.0,
                "[10-20)": 15.0,
                "[20-30)": 25.0,
                "[30-40)": 35.0,
                "[40-50)": 45.0,
                "[50-60)": 55.0,
                "[60-70)": 65.0,
                "[70-80)": 75.0,
                "[80-90)": 85.0,
                "[90-100)": 95.0,
            }
            self.age_num = age_map.get(self.age, 65.0)

        # Rate metrics
        if self.labs_per_day is None:
            self.labs_per_day = round(self.num_lab_procedures / max(1, self.time_in_hospital), 2)
        if self.meds_per_day is None:
            self.meds_per_day = round(self.num_medications / max(1, self.time_in_hospital), 2)

        # Prior visits
        if self.total_prior_visits is None:
            self.total_prior_visits = (
                self.number_inpatient + self.number_emergency + self.number_outpatient
            )

        # Clinical flags
        if self.polypharmacy is None:
            self.polypharmacy = 1 if self.num_medications > 16 else 0
        if self.a1c_tested is None:
            self.a1c_tested = 0 if self.A1Cresult == "None" else 1

        # Medication count
        med_attrs = [
            self.metformin,
            self.repaglinide,
            self.nateglinide,
            self.glimepiride,
            self.glipizide,
            self.glyburide,
            self.pioglitazone,
            self.rosiglitazone,
            self.acarbose,
            self.glyburide_metformin,
            self.insulin,
        ]
        if self.n_active_drugs is None:
            self.n_active_drugs = sum(1 for m in med_attrs if m in ("Steady", "Up", "Down"))
        if self.n_dose_changes is None:
            self.n_dose_changes = sum(1 for m in med_attrs if m in ("Up", "Down"))

        return self


class SimilarRequest(BaseModel):
    patient_features: PatientEncounterRequest
    k: int = Field(default=20, ge=1, le=50)


class FactorExplanation(BaseModel):
    feature: str
    value: Any
    shap_value: float
    explanation: str


class PredictResponse(BaseModel):
    calibrated_risk_probability: float
    risk_tier: str
    decision_threshold: float
    predicted_readmission: bool
    capacity_thresholds: Dict[str, float]
    top_risk_increasing_factors: List[FactorExplanation]
    top_risk_reducing_factors: List[FactorExplanation]
    similar_patients_summary: Dict[str, Any]


class SimilarEncounter(BaseModel):
    encounter_id: Union[int, str]
    similarity_score: float
    age: str
    gender: str
    time_in_hospital: int
    primary_diagnosis: str
    prior_inpatient: int
    readmitted_30d: int


class SimilarResponse(BaseModel):
    k_requested: int
    neighbor_readmission_rate: float
    similar_encounters: List[SimilarEncounter]


class ExplainResponse(BaseModel):
    top_risk_increasing: List[FactorExplanation]
    top_risk_reducing: List[FactorExplanation]
    all_attributions: Dict[str, float]


class CohortPatient(BaseModel):
    encounter_id: int
    patient_nbr: int
    risk_score: float
    risk_tier: str
    readmitted_30d: int
    age: str
    gender: str
    time_in_hospital: int
    number_inpatient: int
    diag_1_group: str


class CohortResponse(BaseModel):
    mode: str
    capacity_fraction: float
    total_test_encounters: int
    flagged_encounters_count: int
    total_true_readmissions_in_cohort: int
    true_readmissions_caught: int
    precision: float
    recall: float
    lift_vs_random: float
    threshold_at_capacity: float
    top_patients: List[CohortPatient]
