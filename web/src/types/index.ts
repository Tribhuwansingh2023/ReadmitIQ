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
  value: string | number;
  shap_value: number;
  explanation: string;
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

export interface PredictResponse {
  calibrated_risk_probability: number;
  risk_tier: "Critical" | "High" | "Medium" | "Low";
  decision_threshold: number;
  predicted_readmission: boolean;
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

export interface MetricsResponse {
  ablation_table?: {
    validation_models: Array<{
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
    }>;
    paired_rag_difference?: {
      delta_auc: number;
      ci_lower: number;
      ci_upper: number;
      crosses_zero: boolean;
    };
  };
  test_results?: {
    model_name: string;
    eval_timestamp?: string;
    git_commit_hash?: string;
    test_cohort_size: number;
    test_positive_prevalence?: number;
    auc_roc: number;
    auc_ci_95: [number, number];
    frozen_decision_threshold: number;
    f1: number;
    precision: number;
    recall: number;
    brier_score: number;
    ece: number;
    capacity_metrics: {
      top_5pct?: {
        threshold: number;
        precision: number;
        lift: number;
        recall: number;
      };
      top_10pct: {
        threshold: number;
        precision: number;
        lift: number;
        recall: number;
      };
      top_20pct?: {
        threshold: number;
        precision: number;
        lift: number;
        recall: number;
      };
    };
    paired_rag_difference_test?: {
      observed_delta_auc: number;
      ci_95: [number, number];
      crosses_zero: boolean;
    };
    validation_comparison?: {
      val_auc: number;
      val_auc_ci: [number, number];
      auc_generalization_gap: number;
    };
  };
  fairness_audit?: {
    evaluation_split?: string;
    sample_size?: number;
    capacity_threshold_evaluated?: number;
    primary_model_test_auc?: number;
    blinded_model_test_auc?: number;
    auc_delta_blinded_vs_primary?: number;
    audits?: Record<
      string,
      {
        reference_group: string;
        subgroups: Record<
          string,
          {
            sample_size: number;
            reliable: boolean;
            prevalence: number;
            auc_roc: number;
            auc_ci_95: [number, number];
            selection_rate: number;
            tpr: number;
            fpr: number;
            ratio_selection_rate_vs_ref: number;
            ratio_tpr_vs_ref: number;
          }
        >;
      }
    >;
  };
  drift_metrics?: {
    features: Record<
      string,
      {
        psi_val: number;
        psi_test: number;
        status_val: string;
        status_test: string;
      }
    >;
    score_drift: {
      psi_val: number;
      psi_test: number;
    };
    summary: {
      total_features_monitored: number;
      moderate_drift_count: number;
      high_drift_count: number;
    };
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
}
