import React from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  Cpu,
  Database,
  FileCode,
  Gauge,
  Network,
  ShieldCheck,
  Split,
} from "lucide-react";

export const HowItWorksPage: React.FC = () => {
  const pipelineSteps = [
    {
      step: 1,
      title: "Raw Encounter Ingestion & Clinical Cohort Exclusions",
      icon: Database,
      badge: "UCI Diabetes 130-US Hospitals",
      badgeColor: "bg-sky-500/10 text-sky-400 border-sky-500/20",
      description:
        "101,766 total hospital encounters across 71,518 unique diabetic patients (1999–2008). Encounters with discharge disposition to hospice (IDs 13, 14) or in-hospital death (IDs 11, 19, 20, 21) are excluded as clinically ineligible for 30-day care transition outreach, yielding 99,340 eligible encounters.",
    },
    {
      step: 2,
      title: "Patient-Level Disjoint Partitioning (70/15/15)",
      icon: Split,
      badge: "Zero Patient Leakage",
      badgeColor: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      description:
        "Patients with multiple hospitalizations (mean 1.42 visits, up to 40) are partitioned strictly at the patient level (patient_nbr) using stratified group hashing. Train (69,535 rows / 48,988 patients), Validation (14,902 rows / 10,520 patients), and Test (14,903 rows / 10,479 patients) share exactly 0 overlapping patients.",
    },
    {
      step: 3,
      title: "Clinical Feature Engineering & Categorical Collapsing",
      icon: FileCode,
      badge: "141 Encoded Features",
      badgeColor: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
      description:
        "ICD-9 codes are mapped into Clinical Classifications Software (CCS) diagnostic groups (Circulatory, Respiratory, Diabetes, etc.). Infrequent categories (<1% prevalence in train) are collapsed into 'Other'. Numerical features are winsorized at 99.5th percentiles. All encoders are fitted strictly on TRAIN data.",
    },
    {
      step: 4,
      title: "Structured Case-Based Retrieval (RAG Engine)",
      icon: Network,
      badge: "64-dim L2 SVD + FAISS",
      badgeColor: "bg-purple-500/10 text-purple-400 border-purple-500/20",
      description:
        "Clinical feature matrix is projected into a 64-dimensional latent semantic space via TruncatedSVD and L2-normalized. For each query encounter, FAISS IndexFlatIP retrieves top k=20 peer precedents. On training data, an out-of-fold 5-fold StratifiedGroupKFold mechanism strictly excludes the query patient and fold.",
    },
    {
      step: 5,
      title: "Primary Gradient Boosted Classifier (LightGBM)",
      icon: Brain,
      badge: "Frozen Winning Model",
      badgeColor: "bg-sky-500/10 text-sky-400 border-sky-500/20",
      description:
        "LightGBM with native categorical handling trained on all 141 tabular features plus the structured RAG peer readmission rate. Hyperparameters (max_depth=4, learning_rate=0.05, min_child_samples=50) selected via 5-fold GroupKFold validation to optimize AUC-ROC while preventing leaf overfitting.",
    },
    {
      step: 6,
      title: "5-Fold OOF Isotonic Probability Calibration",
      icon: Gauge,
      badge: "ECE = 0.0039",
      badgeColor: "bg-amber-500/10 text-amber-400 border-amber-500/20",
      description:
        "Tree boosting raw logits are miscalibrated (Brier score 0.2201, uncalibrated ECE 34.2%). Non-parametric Isotonic Regression fitted on 5-fold out-of-fold training probabilities aligns predicted risk with true observed incidence, reducing test Brier to 0.0973 and Expected Calibration Error to 0.39%.",
    },
    {
      step: 7,
      title: "Dual-Layer Explainability & Precedent Triangulation",
      icon: Activity,
      badge: "SHAP + Peer Cases",
      badgeColor: "bg-teal-500/10 text-teal-400 border-teal-500/20",
      description:
        "Local predictions are audited via TreeExplainer SHAP feature attributions mapped into plain-language clinical reasons (e.g. '3 prior inpatient admissions raises risk'). The care team simultaneously reviews the top 5 closest historical patient trajectories, their diagnoses, and 30-day outcomes.",
    },
  ];

  const safeguards = [
    {
      title: "Strict Patient Disjointness",
      status: "Verified 0 Overlap",
      desc: "Train (48,988), Val (10,520), and Test (10,479) patient IDs are mutually exclusive across all parquet files and retriever metadata.",
    },
    {
      title: "Out-of-Fold RAG Isolation",
      status: "100% Bitwise Match",
      desc: "Brute-force re-derivation on 200 random training encounters confirmed candidates exclude both the query patient and the OOF fold (<1e-6 diff).",
    },
    {
      title: "Train-Only Pipeline Fitting",
      status: "Invariance Proven",
      desc: "Perturbing validation continuous features by 10x and injecting 100 new unseen categories produced 0.000 change in fitted scaler and SVD parameters.",
    },
    {
      title: "Label Permutation Check",
      status: "AUC = 0.5002",
      desc: "Retraining the model on randomly shuffled training labels yielded validation AUC of 0.5002 (pure random chance, confirming zero label memorization).",
    },
    {
      title: "One-Time Test Evaluation Lock",
      status: "Cryptographically Enforced",
      desc: "The held-out test evaluation was executed exactly once and locked under reports/metrics/test_eval.lock. Re-evaluation is programmatically blocked.",
    },
  ];

  const limitations = [
    {
      title: "Historical Clinical Practice Era (1999–2008)",
      desc: "The UCI dataset reflects clinical practice and coding guidelines from 1999–2008. Newer antidiabetic therapeutics (GLP-1 receptor agonists and SGLT2 inhibitors) are not captured.",
    },
    {
      title: "Single-Network Hospital Censoring",
      desc: "Readmissions are recorded only if the patient returned to one of the 130 participating hospitals in the dataset network. Discharges to outside health systems appear as non-readmissions.",
    },
    {
      title: "Synthetic Clinical Notes Signal",
      desc: "Synthetic notes were systematically generated from tabular attributes. While linguistically realistic, they contain no independent clinical information beyond the structured features.",
    },
    {
      title: "Modest Discrimination (AUC 0.6590)",
      desc: "Predicting 30-day all-cause readmission solely from administrative and diagnostic records without continuous inpatient telemetry, vitals, or outpatient social determinants yields modest AUC (0.64–0.68).",
    },
    {
      title: "Neutral Paired RAG Gain (+0.0004 AUC)",
      desc: "Structured RAG does not offer statistically significant discrimination gain over baseline LightGBM (95% CI [-0.0025, +0.0031] crosses zero). Its primary clinical value is case-based interpretability.",
    },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-sky-500/10 text-sky-400 border border-sky-500/20">
            System Architecture & Methodology
          </span>
          <span className="text-xs text-slate-400">
            End-to-End Pipeline & Governance Specifications
          </span>
        </div>
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2.5">
          <Cpu className="w-6 h-6 text-sky-400" />
          How ReadmitIQ Works
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-3xl">
          An honest, production-ready clinical decision support architecture
          combining gradient boosted trees, case-based latent retrieval (RAG),
          isotonic probability calibration, and local SHAP factor attribution.
        </p>
      </div>

      {/* Pipeline Diagram Walkthrough */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Network className="w-5 h-5 text-indigo-400" />
            7-Stage Clinical ML Pipeline
          </h3>
          <span className="text-xs text-slate-500 font-mono">
            Sequential Execution Flow
          </span>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {pipelineSteps.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.step}
                className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all flex flex-col md:flex-row md:items-start gap-4 shadow-md"
              >
                <div className="flex items-center gap-3 md:flex-col md:items-center">
                  <div className="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700/80 flex items-center justify-center flex-shrink-0 text-white font-mono font-bold text-sm">
                    {s.step}
                  </div>
                  <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400 flex-shrink-0 hidden md:flex">
                    <Icon className="w-4 h-4" />
                  </div>
                </div>

                <div className="flex-1 space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h4 className="text-base font-semibold text-white">
                      {s.title}
                    </h4>
                    <span
                      className={`text-[11px] font-mono px-2 py-0.5 rounded-full border ${s.badgeColor}`}
                    >
                      {s.badge}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {s.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Safeguards Grid */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              Cryptographic & Algorithmic Leakage Safeguards
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Strict isolation preventing target, patient, and temporal leakage
              across all stages
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            Adversarial Audit: PASS
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {safeguards.map((sg, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white">
                  {sg.title}
                </span>
                <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  {sg.status}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {sg.desc}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Honest Limitations Card */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              Honest Clinical & Methodological Limitations
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Transparent real-world caveats required for ethical medical AI
              deployment
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
            Clinical Governance
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {limitations.map((lim, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-1.5"
            >
              <span className="text-xs font-semibold text-amber-300 block">
                {lim.title}
              </span>
              <p className="text-xs text-slate-400 leading-relaxed">
                {lim.desc}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
