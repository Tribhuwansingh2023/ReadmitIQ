export interface HealthResponse {
  status: string;
  version: string;
  git_commit_hash: string;
  frozen_timestamp?: string;
  model_name?: string;
  artifacts_loaded: Record<string, boolean>;
}

export interface FactorExplanation {
  feature: string;
  value?: string | number;
  shap_value: number;
  explanation: string;
  feature_name?: string;
  attribution?: number;
  clinical_meaning?: string;
}

export interface SimilarEncounter {
  encounter_id: number | string;
  similarity_score: number;
  age: string;
  gender: string;
  time_in_hospital: number;
  primary_diagnosis: string;
  prior_inpatient: number;
  readmitted_30d: number;
}

export interface SuggestedIntervention {
  id: string;
  action: string;
  category: string;
  priority: "High" | "Medium" | "Low";
  trigger_driver: string;
  rationale: string;
  target_timeline: string;
}

export interface SuggestedInterventionsSummary {
  disclaimer: string;
  risk_tier: string;
  suggestions_count: number;
  suggestions: SuggestedIntervention[];
}

export interface PredictResponse {
  calibrated_risk_probability: number;
  risk_tier: "Critical" | "High" | "Medium" | "Low";
  decision_threshold: number;
  predicted_readmission: boolean;
  raw_score?: number;
  calibrated_prob?: number;
  tier?: "Critical" | "High" | "Medium" | "Low";
  capacity_thresholds: {
    top_5pct_critical: number;
    top_10pct_high: number;
    top_20pct_medium: number;
  };
  top_risk_increasing_factors: FactorExplanation[];
  top_risk_reducing_factors: FactorExplanation[];
  similar_patients_summary: {
    k_neighbors: number;
    neighbor_readmission_rate: number;
    top_neighbors: SimilarEncounter[];
  };
  suggested_interventions?: SuggestedInterventionsSummary;
}

export interface CohortPatient {
  encounter_id: number;
  patient_nbr: number;
  risk_score: number;
  risk_tier: string;
  readmitted_30d: number;
  age: string;
  gender: string;
  time_in_hospital: number;
  number_inpatient: number;
  diag_1_group: string;
}

export interface SimilarResponse {
  k_requested: number;
  neighbor_readmission_rate: number;
  similar_encounters: SimilarEncounter[];
}

export interface ExplainResponse {
  top_risk_increasing: FactorExplanation[];
  top_risk_reducing: FactorExplanation[];
  all_attributions: Record<string, number>;
}

export interface CohortResponse {
  mode: string;
  capacity_fraction: number;
  total_test_encounters: number;
  flagged_encounters_count: number;
  total_true_readmissions_in_cohort: number;
  true_readmissions_caught: number;
  precision: number;
  recall: number;
  lift_vs_random: number;
  threshold_at_capacity: number;
  top_patients: CohortPatient[];
}

export interface ValidationModelItem {
  model: string;
  feature_set: string;
  auc: number;
  auc_ci_lower?: number;
  auc_ci_upper?: number;
  brier_score?: number;
  ece?: number;
  precision_at_top10?: number;
  lift_at_top10?: number;
  optimal_threshold?: number;
  f1_at_optimal_threshold?: number;
  paired_delta_auc?: number;
  paired_ci_95?: [number, number];
  ci_crosses_zero?: boolean;
}

export interface MetricsResponse {
  ablation_table?: {
    transformer_status?: string;
    transformer_pending_message?: string;
    validation_models?: ValidationModelItem[];
    ablation_results?: any[];
    paired_rag_difference?: any;
    paired_bootstrap_differences?: any;
    calibration_report?: any;
  };
  test_results?: {
    model_name?: string;
    eval_timestamp?: string;
    git_commit_hash?: string;
    test_cohort_size?: number;
    test_positive_prevalence?: number;
    auc_roc?: number;
    auc_ci_95?: [number, number];
    frozen_decision_threshold?: number;
    f1?: number;
    precision?: number;
    recall?: number;
    brier_score?: number;
    ece?: number;
    paired_rag_difference_test?: any;
    validation_comparison?: any;
    test_metrics?: {
      auc_roc: number;
      auc_ci_95: [number, number];
      brier_score: number;
      ece: number;
      constant_prevalence_brier?: number;
    };
    capacity_metrics?: {
      top_5pct?: { threshold: number; precision: number; lift: number; recall: number };
      top_10pct?: { threshold: number; precision: number; lift: number; recall: number };
      top_20pct?: { threshold: number; precision: number; lift: number; recall: number };
    };
    calibration_curve?: {
      prob_true: number[];
      prob_pred: number[];
      counts: number[];
    };
  };
  fairness_audit?: {
    demographic_parity?: Record<string, any>;
    equalized_odds?: Record<string, any>;
    audits?: Record<string, any>;
    [key: string]: any;
  };
  drift_metrics?: {
    features?: Record<string, any>;
    scores?: Record<string, any>;
    summary?: Record<string, any>;
  };
  frozen_selection?: any;
}

export interface PatientFormInput {
  age: string;
  gender: string;
  race: string;
  time_in_hospital: number;
  num_lab_procedures: number;
  num_procedures: number;
  num_medications: number;
  number_outpatient: number;
  number_emergency: number;
  number_inpatient: number;
  number_diagnoses: number;
  admission_type_group: string;
  admission_source_group: string;
  discharge_disposition_group: string;
  medical_specialty: string;
  payer_code: string;
  diag_1_group: string;
  diag_2_group: string;
  diag_3_group: string;
  max_glu_serum: string;
  A1Cresult: string;
  diabetesMed: string;
  change: string;
  insulin: string;
  metformin: string;
  glipizide?: string;
  glyburide?: string;
  glimepiride?: string;
  pioglitazone?: string;
  rosiglitazone?: string;
  acarbose?: string;
  repaglinide?: string;
  nateglinide?: string;
  "glyburide-metformin"?: string;
  external_ref?: string;
  source?: string;
}

// ---------------------------------------------------------------------------
// v1 Clinical Workflow & Desk Types
// ---------------------------------------------------------------------------

export type RiskTier = "Critical" | "High" | "Medium" | "Low";
export type EncounterStatus = "new" | "assigned" | "contacted" | "followup_scheduled" | "closed";

export interface WorklistItem {
  id: number;
  received_at: string;
  source: "manual" | "csv" | "api" | "simulated_feed";
  external_ref?: string;
  raw_score: number;
  calibrated_prob: number;
  tier: RiskTier;
  status: EncounterStatus;
  assignee?: string;
  updated_at: string;
  outcome_readmitted_30d?: number | null;
  patient_summary: {
    age: string;
    gender: string;
    time_in_hospital: number;
    primary_diagnosis: string;
    number_inpatient: number;
  };
}

export interface WorklistResponse {
  mode: "standard" | "capacity_mode";
  capacity_limit?: number;
  total_count: number;
  tier_counts: Record<string, number>;
  page?: number;
  page_size?: number;
  items: WorklistItem[];
}

export interface DischargeDetailResponse {
  id: number;
  received_at: string;
  source: string;
  external_ref?: string;
  raw_score: number;
  calibrated_prob: number;
  tier: RiskTier;
  status: EncounterStatus;
  assignee?: string;
  model_version: string;
  outcome_readmitted_30d?: number | null;
  outcome_at?: string | null;
  features: PatientFormInput;
  shap_explanation: {
    top_risk_increasing: FactorExplanation[];
    top_risk_reducing: FactorExplanation[];
    all_attributions: Record<string, number>;
  };
  similar_patients_summary?: {
    k_neighbors: number;
    neighbor_readmission_rate: number;
    top_neighbors: SimilarEncounter[];
  };
  suggested_interventions?: SuggestedInterventionsSummary;
  action_history: {
    id: number;
    type: string;
    note: string;
    created_at: string;
    actor: string;
  }[];
  audit_trail: {
    id: number;
    ts: string;
    actor: string;
    event: string;
    payload?: any;
  }[];
}

export interface FeedStatusResponse {
  is_running: boolean;
  rate_per_minute: number;
  seed: number;
  total_emitted: number;
  simulated_time: string;
  feed_source: string;
}

export interface MonitoringResponse {
  timestamp: string;
  counts: {
    total_discharges: number;
    simulated_feed_count: number;
    api_count: number;
    csv_count: number;
    open_worklist_count: number;
    resolved_outcomes_count: number;
    positive_readmissions_count: number;
  };
  tier_mix: Record<
    string,
    { count: number; share: number; expected_share: number; delta: number }
  >;
  score_drift: {
    metric: string;
    raw_score_psi: number;
    severity: string;
    sample_size: number;
    thresholds: Record<string, string>;
  };
  data_quality: {
    conformity_rate: number;
    missing_rate: number;
    status: string;
  };
  outcomes_evaluation: {
    insufficient_outcomes: boolean;
    sample_size_outcomes: number;
    positive_readmissions: number;
    minimum_required_outcomes: number;
    minimum_required_positives: number;
    status_message?: string;
    auc_roc?: number;
    auc_ci_95?: [number, number];
    frozen_val_auc_ci?: [number, number];
    precision_at_capacity?: Record<
      string,
      { k_encounters: number; caught_readmissions: number; precision: number }
    >;
    calibration_deciles?: {
      bin_range: string;
      count: number;
      predicted_prob_mean: number;
      observed_readmission_rate: number;
    }[];
  };
  alerts: { severity: string; type: string; message: string }[];
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  git_commit_hash: string;
  frozen_timestamp?: string;
  serving_thresholds?: {
    top_5pct: number;
    top_10pct: number;
    top_20pct: number;
  };
  tier_definitions?: Record<string, string>;
  splits_summary?: Record<string, { encounters: number; strategy: string }>;
  model_card_url: string;
  api_contract_url: string;
}
