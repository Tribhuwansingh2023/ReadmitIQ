import type { StatProvenance } from "./Stat";

export const PROVENANCES: Record<string, StatProvenance> = {
  // --- Desk headline counters ---
  activeWorklistCount: {
    source: "GET /v1/worklist (discharges table)",
    howComputed: "Count of all active inpatient discharges awaiting post-discharge review",
    verifiedArtifactPath: "models/final/serving_thresholds.json",
  },
  criticalOpenCount: {
    source: "GET /v1/worklist?tier=Critical&status=open",
    howComputed: "Encounter count in top 5% raw score tier (raw_score >= 0.683933) with status != 'closed'",
    verifiedArtifactPath: "models/final/serving_thresholds.json",
  },
  contactedCount: {
    source: "GET /v1/worklist (actions table)",
    howComputed: "Discharges with recorded clinical contact or completed medication reconciliation",
    verifiedArtifactPath: "api/database.py",
  },
  resolvedOutcomesCount: {
    source: "GET /v1/monitoring (discharges.outcome_readmitted_30d)",
    howComputed: "Released 30-day readmission status verified through hospital follow-up or simulated time-machine",
    verifiedArtifactPath: "api/feed.py",
  },

  // --- Frozen Model Serving Thresholds ---
  criticalThreshold: {
    source: "models/final/serving_thresholds.json",
    howComputed: "95th percentile (top 5%) raw LightGBM score on frozen validation split (N=14,902)",
    ci: "Point estimate: 0.683933",
    verifiedArtifactPath: "models/final/serving_thresholds.json",
  },
  highThreshold: {
    source: "models/final/serving_thresholds.json",
    howComputed: "90th percentile (top 10%) raw LightGBM score on frozen validation split (N=14,902)",
    ci: "Point estimate: 0.640775",
    verifiedArtifactPath: "models/final/serving_thresholds.json",
  },
  mediumThreshold: {
    source: "models/final/serving_thresholds.json",
    howComputed: "80th percentile (top 20%) raw LightGBM score on frozen validation split (N=14,902)",
    ci: "Point estimate: 0.579340",
    verifiedArtifactPath: "models/final/serving_thresholds.json",
  },

  // --- Offline Evidence & Test Split Benchmarks ---
  testSplitAuc: {
    source: "reports/metrics/test_results.json",
    howComputed: "ROC AUC evaluated once on held-out test split (N=14,903, 1,698 positives) with 1,000 bootstrap resamples",
    ci: "95% CI: [0.6508, 0.6780]",
    verifiedArtifactPath: "reports/metrics/test_results.json",
  },
  testPrecisionTop10: {
    source: "reports/metrics/test_results.json",
    howComputed: "True readmissions caught among top 10% highest risk encounters (k=1,490)",
    ci: "Precision: 23.76% (2.09x lift over 11.37% baseline)",
    verifiedArtifactPath: "reports/metrics/test_results.json",
  },
  testBrierScore: {
    source: "reports/metrics/test_results.json",
    howComputed: "Mean squared error of calibrated risk probabilities vs binary 30-day outcomes",
    ci: "Brier: 0.0973 (Constant-prevalence null reference: 0.1008)",
    verifiedArtifactPath: "reports/metrics/test_results.json",
  },
  testEce: {
    source: "reports/metrics/test_results.json",
    howComputed: "Expected Calibration Error across 10 uniform reliability probability bins",
    ci: "ECE: 0.0039",
    verifiedArtifactPath: "reports/metrics/test_results.json",
  },

  // --- Monitoring Drift & Calibration ---
  scoreDriftPsi: {
    source: "GET /v1/monitoring (raw_score_psi)",
    howComputed: "Population Stability Index computed across 10 training decile reference bins (PSI < 0.10 stable)",
    verifiedArtifactPath: "models/final/serving_thresholds.json",
  },
};
