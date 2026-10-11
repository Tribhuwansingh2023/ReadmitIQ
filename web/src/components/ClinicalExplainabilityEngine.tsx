import React, { useState, useEffect } from "react";
import {
  TrendingUp,
  TrendingDown,
  Info,
  ShieldAlert,
  CheckCircle2,
  Activity,
  Layers,
  Scale,
  Users,
  Sliders,
  HelpCircle,
} from "lucide-react";
import { apiClient } from "../api/client";
import { Stat } from "./Stat";
import { PROVENANCES } from "./Provenances";
import type {
  FactorExplanation,
  ShapExplanationPayload,
  SimilarEncounter,
} from "../types";

interface ClinicalExplainabilityEngineProps {
  encounterId?: string | number;
  calibratedProb: number;
  rawScore?: number;
  rawMargin?: number;
  baseValue?: number;
  riskTier: string;
  modelVersion?: string;
  shapExplanation?: ShapExplanationPayload;
  explanationV2?: any;
  similarEncounters?: SimilarEncounter[];
  patientFeatures?: Record<string, any>;
  compact?: boolean;
}

export const ClinicalExplainabilityEngine: React.FC<
  ClinicalExplainabilityEngineProps
> = ({
  calibratedProb,
  riskTier,
  modelVersion = "LightGBM v1.0.0 (Frozen)",
  shapExplanation,
  explanationV2,
}) => {
  // Navigation tabs matching clinical decision workflow
  const [activeTab, setActiveTab] = useState<
    "score" | "tier" | "change" | "evidence" | "global"
  >("score");
  const [showHowToRead, setShowHowToRead] = useState<boolean>(false);

  // Global model explanation state
  const [globalData, setGlobalData] = useState<any | null>(null);
  const [globalLoading, setGlobalLoading] = useState<boolean>(false);

  useEffect(() => {
    if (activeTab === "global" && !globalData && !globalLoading) {
      setGlobalLoading(true);
      apiClient
        .getXaiGlobal()
        .then((res) => {
          setGlobalData(res);
        })
        .catch(() => {
          // fallback
          apiClient.getGlobalExplanation().then(setGlobalData).catch(() => {});
        })
        .finally(() => {
          setGlobalLoading(false);
        });
    }
  }, [activeTab, globalData, globalLoading]);

  // Extract V2 data if present, otherwise construct compatible structure
  const v2 = explanationV2;
  const waterfall = v2?.calibrated_waterfall;
  const contrastive = v2?.contrastive_explanation;
  const counterfactual = v2?.counterfactual;
  const groupAttributions = v2?.group_attributions || {};
  const unrecordedItems = v2?.information_not_recorded || [];
  const evidenceAgreement = v2?.evidence_agreement;
  const localReliability = v2?.local_reliability;

  // Fallback drivers
  const increasingFactors: FactorExplanation[] =
    v2?.top_risk_increasing || shapExplanation?.top_risk_increasing || [];
  const reducingFactors: FactorExplanation[] =
    v2?.top_risk_reducing || shapExplanation?.top_risk_reducing || [];

  return (
    <div className="bg-surface rounded-sm hairline-border space-y-4 font-sans text-primary">
      {/* Header bar with clinical guidance & limits popover toggle */}
      <div className="p-4 sm:p-5 border-b border-hairline flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-canvas/40">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-subtle text-muted font-bold">
              XAI Clinical Layer v2
            </span>
            <span className="text-xs text-muted font-mono">
              Additivity Asserted &bull; Empirical Reliability Validated
            </span>
          </div>
          <h3 className="text-base font-semibold text-primary mt-1">
            Clinical Explanation & Confidence Dashboard
          </h3>
          <p className="text-xs text-muted mt-0.5">
            Transparent attribution of model log-odds margin, calibrated probability steps, and precedent peer consistency.
          </p>
        </div>

        <button
          onClick={() => setShowHowToRead(!showHowToRead)}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium rounded border border-hairline bg-surface hover:bg-subtle text-primary transition-colors self-start sm:self-auto"
          aria-expanded={showHowToRead}
          aria-label="Toggle clinical interpretation guidance"
        >
          <HelpCircle className="w-3.5 h-3.5 text-accent" />
          <span>How to read this (Honesty Rules)</span>
        </button>
      </div>

      {/* "How to read this" Honest Guidance Popover */}
      {showHowToRead && (
        <div className="mx-4 sm:mx-5 p-4 bg-canvas border border-accent/30 rounded-sm text-xs space-y-3 font-sans animate-fade-in">
          <div className="flex items-center justify-between border-b border-hairline pb-2">
            <div className="flex items-center gap-2 text-accent font-semibold">
              <Info className="w-4 h-4" />
              <span>Mandatory Clinical Interpretation & Governance Principles</span>
            </div>
            <button
              onClick={() => setShowHowToRead(false)}
              className="text-muted hover:text-primary text-[11px] font-mono"
            >
              Close [&times;]
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-muted leading-relaxed">
            <div className="space-y-1.5">
              <p>
                <strong className="text-primary font-mono">1. Associative, Not Causal:</strong> TreeSHAP values measure statistical associations in the model's learned distribution. They do not demonstrate causal clinical mechanisms or therapeutic treatment efficacy.
              </p>
              <p>
                <strong className="text-primary font-mono">2. Log-Odds Margin vs Calibrated Risk:</strong> Feature attributions operate strictly on the raw model margin (log-odds). Displayed readmission probabilities are produced via monotonic isotonic calibration.
              </p>
            </div>
            <div className="space-y-1.5">
              <p>
                <strong className="text-primary font-mono">3. Information Not Recorded:</strong> Missing administrative categories and unperformed labs are segregated from clinical drivers. Missing data is never classified as a clinical pathology or risk factor.
              </p>
              <p>
                <strong className="text-primary font-mono">4. Decision Support Only:</strong> Outputs augment clinical transition-of-care workflows and must never supersede multidisciplinary physician evaluation.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="px-4 sm:px-5 border-b border-hairline flex items-center gap-4 text-xs font-mono overflow-x-auto">
        {[
          { key: "score", label: "Why this score" },
          { key: "tier", label: "Why this tier" },
          { key: "change", label: "What can change" },
          { key: "evidence", label: "Evidence & Reliability" },
          { key: "global", label: "Global Model Knowledge" },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`py-2.5 border-b-2 font-medium transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              activeTab === tab.key
                ? "border-accent text-accent font-bold"
                : "border-transparent text-muted hover:text-primary"
            }`}
            aria-selected={activeTab === tab.key}
            role="tab"
          >
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* TAB 1: WHY THIS SCORE (Waterfall, Groups, Top Drivers, Unrecorded) */}
      {activeTab === "score" && (
        <div className="p-4 sm:p-5 space-y-6">
          {/* Calibrated Probability Waterfall Decomposition */}
          {waterfall && waterfall.steps && (
            <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-hairline pb-2.5">
                <div>
                  <h4 className="text-xs font-bold text-primary uppercase font-mono tracking-wider flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-accent" />
                    <span>Calibrated Probability Waterfall (Order-Dependent Decomposition)</span>
                  </h4>
                  <p className="text-[11px] text-muted">
                    Accumulates margin from reference score through non-linear isotonic calibration into calibrated percentage points.
                  </p>
                </div>
                <div className="text-right font-mono text-[11px] text-muted">
                  <span>Reference Base: {(waterfall.reference_risk_at_base_margin * 100).toFixed(1)}%</span>
                  <span className="mx-1.5">&bull;</span>
                  <span className="text-primary font-bold">Final Risk: {(waterfall.final_calibrated_prob * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Waterfall Steps Bar List */}
              <div className="space-y-2 pt-1 font-mono text-xs">
                {waterfall.steps.map((st: any, idx: number) => {
                  const delta = st.calibrated_prob_step_delta * 100;
                  const isPos = delta >= 0.001;
                  const isNeg = delta <= -0.001;
                  return (
                    <div
                      key={idx}
                      className="p-2.5 bg-surface hairline-border rounded flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-muted font-mono w-4">#{idx + 1}</span>
                        <span className="font-semibold text-primary">{st.feature_name}</span>
                        <span className="text-[10px] text-muted">({st.shap_margin > 0 ? `+${st.shap_margin.toFixed(3)}` : st.shap_margin.toFixed(3)} log-odds)</span>
                      </div>
                      <div className="flex items-center gap-4 self-end sm:self-auto">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                            isPos
                              ? "text-critical bg-critical/10"
                              : isNeg
                              ? "text-protective bg-protective/10"
                              : "text-muted bg-subtle"
                          }`}
                        >
                          {delta > 0 ? `+${delta.toFixed(2)}% pts` : `${delta.toFixed(2)}% pts`}
                        </span>
                        <span className="text-muted text-[11px]">
                          Cumulative: <strong className="text-primary">{(st.calibrated_prob_cumulative * 100).toFixed(1)}%</strong>
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Grouped Clinical Domain Attributions */}
          {Object.keys(groupAttributions).length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-hairline pb-2">
                <h4 className="text-xs font-bold text-primary uppercase font-mono tracking-wider flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-accent" />
                  <span>Clinical Domain Group Net Attributions</span>
                </h4>
                <span className="text-[10px] font-mono text-muted">9 Clinical Domains</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(groupAttributions).map(([gKey, gVal]: [string, any]) => {
                  const isPos = gVal.net_attribution_margin > 0.001;
                  const isNeg = gVal.net_attribution_margin < -0.001;
                  return (
                    <div key={gKey} className="p-3 bg-canvas hairline-border rounded-sm space-y-1.5">
                      <div className="flex items-start justify-between gap-1">
                        <span className="font-semibold text-xs text-primary">{gVal.group_name}</span>
                        <span
                          className={`font-mono text-[10px] px-1.5 py-0.5 rounded font-bold ${
                            isPos
                              ? "text-critical bg-critical/10"
                              : isNeg
                              ? "text-protective bg-protective/10"
                              : "text-muted bg-subtle"
                          }`}
                        >
                          {gVal.net_attribution_margin > 0 ? `+${gVal.net_attribution_margin.toFixed(3)}` : gVal.net_attribution_margin.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-[10px] text-muted font-sans leading-tight line-clamp-2">
                        {gVal.group_rationale}
                      </p>
                      <div className="flex items-center justify-between text-[10px] font-mono text-muted pt-1 border-t border-hairline/60">
                        <span>{gVal.feature_count} features</span>
                        <span>{gVal.share_pct.toFixed(1)}% attribution</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Top Recorded Drivers Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Increasing Drivers */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-critical font-semibold text-xs font-mono uppercase tracking-wider">
                <TrendingUp className="w-4 h-4" />
                <span>Top Risk Increasing Clinical Drivers</span>
              </div>
              <div className="space-y-2">
                {increasingFactors.map((f: any, idx: number) => {
                  const sVal = f.shap_value ?? f.attribution ?? 0;
                  const strength = f.strength || (Math.abs(sVal) > 0.02 ? "large" : Math.abs(sVal) > 0.003 ? "moderate" : "small");
                  return (
                    <div key={idx} className="p-3 bg-canvas hairline-border rounded-sm space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs text-primary">{f.feature_name || f.feature}</span>
                        <div className="flex items-center gap-1.5">
                          <span className={`text-[10px] font-mono uppercase px-1.5 py-0.2 rounded font-bold ${
                            strength === "large" ? "bg-critical/20 text-critical" : strength === "moderate" ? "bg-caution/20 text-caution" : "bg-subtle text-muted"
                          }`}>
                            {strength}
                          </span>
                          <span className="font-mono text-xs font-bold text-critical">+{sVal.toFixed(4)}</span>
                        </div>
                      </div>
                      <div className="text-[11px] text-muted flex items-center justify-between font-mono">
                        <span>Patient Value: <strong className="text-primary">{f.value !== undefined ? String(f.value) : "—"}</strong></span>
                        {f.share_of_attribution_pct && <span>{f.share_of_attribution_pct}% share</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Reducing Drivers */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-protective font-semibold text-xs font-mono uppercase tracking-wider">
                <TrendingDown className="w-4 h-4" />
                <span>Top Risk Reducing Protective Factors</span>
              </div>
              <div className="space-y-2">
                {reducingFactors.map((f: any, idx: number) => {
                  const sVal = f.shap_value ?? f.attribution ?? 0;
                  const strength = f.strength || (Math.abs(sVal) > 0.02 ? "large" : Math.abs(sVal) > 0.003 ? "moderate" : "small");
                  return (
                    <div key={idx} className="p-3 bg-canvas hairline-border rounded-sm space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs text-primary">{f.feature_name || f.feature}</span>
                        <div className="flex items-center gap-1.5">
                          <span className={`text-[10px] font-mono uppercase px-1.5 py-0.2 rounded font-bold ${
                            strength === "large" ? "bg-protective/20 text-protective" : strength === "moderate" ? "bg-caution/20 text-caution" : "bg-subtle text-muted"
                          }`}>
                            {strength}
                          </span>
                          <span className="font-mono text-xs font-bold text-protective">{sVal.toFixed(4)}</span>
                        </div>
                      </div>
                      <div className="text-[11px] text-muted flex items-center justify-between font-mono">
                        <span>Patient Value: <strong className="text-primary">{f.value !== undefined ? String(f.value) : "—"}</strong></span>
                        {f.share_of_attribution_pct && <span>{f.share_of_attribution_pct}% share</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Segregated Information Not Recorded Section */}
          {unrecordedItems.length > 0 && (
            <div className="p-4 bg-canvas hairline-border rounded-sm space-y-2.5">
              <div className="flex items-center gap-2 text-muted">
                <Info className="w-4 h-4 text-accent" />
                <h4 className="text-xs font-bold uppercase font-mono tracking-wider text-primary">
                  Information Not Recorded (Clinical Data Completeness)
                </h4>
              </div>
              <p className="text-[11px] text-muted leading-relaxed">
                The following attributes were unrecorded, omitted, or unmeasured during this encounter. Under ReadmitIQ honesty rules, missing administrative categories and unperformed labs are segregated and never classified as clinical risk factors.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
                {unrecordedItems.map((item: any, idx: number) => (
                  <div key={idx} className="p-2.5 bg-surface hairline-border rounded text-xs space-y-1 font-mono">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-primary">{item.feature_name}</span>
                      <span className="text-[10px] text-muted uppercase bg-subtle px-1.5 py-0.5 rounded">Unrecorded</span>
                    </div>
                    <p className="text-[11px] text-muted font-sans leading-tight">
                      {item.neutral_text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: WHY THIS TIER (Contrastive Explanation) */}
      {activeTab === "tier" && (
        <div className="p-4 sm:p-5 space-y-4">
          <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
            <div className="flex items-center justify-between border-b border-hairline pb-2.5">
              <div>
                <h4 className="text-xs font-bold text-primary uppercase font-mono tracking-wider flex items-center gap-1.5">
                  <Scale className="w-3.5 h-3.5 text-accent" />
                  <span>Contrastive Analysis: Why {contrastive?.current_tier || riskTier} and not {contrastive?.compared_tier || "Next Lower Tier"}?</span>
                </h4>
                <p className="text-[11px] text-muted">
                  Compares this patient's feature attributions against the median patient profile in the {contrastive?.compared_tier || "lower"} validation cohort.
                </p>
              </div>
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-accent/15 text-accent">
                {contrastive?.current_tier || riskTier} Tier
              </span>
            </div>

            <p className="text-xs text-muted leading-relaxed font-sans">
              {contrastive?.comparison_rationale ||
                `Isolating key clinical drivers responsible for pushing this patient into the ${riskTier} risk bracket rather than the next lower threshold.`}
            </p>

            {/* Gap Table */}
            {contrastive?.top_gap_features && contrastive.top_gap_features.length > 0 ? (
              <div className="border border-hairline rounded overflow-hidden pt-1">
                <table className="w-full text-left text-xs font-sans">
                  <thead className="bg-subtle text-[11px] font-mono text-muted uppercase border-b border-hairline">
                    <tr>
                      <th className="p-2.5">Feature</th>
                      <th className="p-2.5 text-right">Patient SHAP</th>
                      <th className="p-2.5 text-right">Lower Tier Median</th>
                      <th className="p-2.5 text-right font-bold text-primary">Attribution Gap (&Delta;)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-hairline font-mono text-[11px]">
                    {contrastive.top_gap_features.map((gap: any, idx: number) => (
                      <tr key={idx} className="hover:bg-subtle/40">
                        <td className="p-2.5 font-medium text-primary">{gap.feature_name}</td>
                        <td className="p-2.5 text-right text-muted">{gap.patient_shap.toFixed(4)}</td>
                        <td className="p-2.5 text-right text-muted">{gap.lower_tier_median_shap.toFixed(4)}</td>
                        <td className="p-2.5 text-right font-bold text-critical">+{gap.shap_gap.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-4 text-center text-muted text-xs font-mono">
                Patient is in the baseline tier; no lower comparison tier applies.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: WHAT CAN CHANGE (Modifiable Levers & Counterfactuals) */}
      {activeTab === "change" && (
        <div className="p-4 sm:p-5 space-y-4">
          <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
            <div className="flex items-center justify-between border-b border-hairline pb-2.5">
              <div className="flex items-center gap-1.5 text-accent font-bold text-xs uppercase font-mono tracking-wider">
                <Sliders className="w-4 h-4" />
                <span>Constrained Modifiable Counterfactual Simulation</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-subtle text-muted">
                Training-Observed Levers Only
              </span>
            </div>

            <p className="text-xs text-muted leading-relaxed font-sans">
              Greedy search over modifiable discharge levers (discharge destination, medication modifications, and inpatient glycemic testing) to evaluate whether realistic clinical adjustments can shift the patient into a lower risk tier.
            </p>

            {counterfactual?.feasible_change_found ? (
              <div className="p-3.5 bg-surface hairline-border rounded space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-protective flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Feasible Care Transition Levers Identified</span>
                  </span>
                  <span className="text-xs font-mono font-bold text-primary">
                    {counterfactual.tier_before} &rarr; <span className="text-protective">{counterfactual.tier_after}</span> ({counterfactual.calibrated_prob_before * 100}% &rarr; {(counterfactual.calibrated_prob_after * 100).toFixed(1)}%)
                  </span>
                </div>

                <div className="space-y-2 pt-1 font-mono text-xs">
                  {counterfactual.modifications.map((mod: any, idx: number) => (
                    <div key={idx} className="p-2.5 bg-canvas hairline-border rounded flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div>
                        <span className="font-semibold text-primary">{mod.feature_name}: </span>
                        <span className="text-muted line-through mr-1.5">{String(mod.from_value)}</span>
                        <span className="text-accent font-bold">&rarr; {String(mod.to_value)}</span>
                      </div>
                      <span className="text-[11px] text-muted font-sans">{mod.rationale}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-3 bg-subtle/50 hairline-border rounded text-xs font-mono text-muted space-y-1">
                <span className="font-semibold text-primary">Simulation Result:</span>
                <p>{counterfactual?.message || "No combination of up to 2 training-observed modifiable adjustments lowered the predicted risk tier."}</p>
              </div>
            )}

            <div className="p-2.5 bg-subtle/30 rounded text-[11px] text-muted italic font-mono border-t border-hairline/60">
              Disclaimer: {counterfactual?.disclaimer || "Associative sensitivity simulation, not a causal treatment effect."}
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: EVIDENCE & EMPIRICAL RELIABILITY */}
      {activeTab === "evidence" && (
        <div className="p-4 sm:p-5 space-y-4">
          {/* Precedent Retrieval Evidence Agreement */}
          {evidenceAgreement && (
            <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
              <div className="flex items-center justify-between border-b border-hairline pb-2.5">
                <div className="flex items-center gap-1.5 text-primary font-bold text-xs uppercase font-mono tracking-wider">
                  <Users className="w-4 h-4 text-accent" />
                  <span>Retrieval Evidence Agreement (FAISS Peer Consistency)</span>
                </div>
                <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded uppercase ${
                  evidenceAgreement.agreement_status === "consistent"
                    ? "bg-protective/15 text-protective"
                    : evidenceAgreement.agreement_status === "inconsistent"
                    ? "bg-critical/15 text-critical"
                    : "bg-caution/15 text-caution"
                }`}>
                  {evidenceAgreement.agreement_status}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-2.5 bg-surface hairline-border rounded">
                  <span className="text-muted text-[10px] block">Model Calibrated Risk</span>
                  <span className="text-primary font-bold text-base">{(calibratedProb * 100).toFixed(1)}%</span>
                </div>
                <div className="p-2.5 bg-surface hairline-border rounded">
                  <span className="text-muted text-[10px] block">Peer Precedent Readmission Rate (k=20)</span>
                  <span className="text-primary font-bold text-base">
                    {evidenceAgreement.peer_readmission_rate ? `${(evidenceAgreement.peer_readmission_rate * 100).toFixed(1)}%` : "—"}
                  </span>
                  {evidenceAgreement.peer_wilson_ci_95 && (
                    <span className="text-[10px] text-muted block">
                      Wilson 95% CI: [{(evidenceAgreement.peer_wilson_ci_95[0] * 100).toFixed(1)}%, {(evidenceAgreement.peer_wilson_ci_95[1] * 100).toFixed(1)}%]
                    </span>
                  )}
                </div>
              </div>

              <p className="text-xs text-muted font-sans leading-relaxed">
                {evidenceAgreement.rule_evaluation}
              </p>
            </div>
          )}

          {/* Local Validation Reliability Statement */}
          {localReliability && (
            <div className="p-4 bg-canvas hairline-border rounded-sm space-y-2.5">
              <div className="flex items-center gap-1.5 text-primary font-bold text-xs uppercase font-mono tracking-wider">
                <ShieldAlert className="w-4 h-4 text-accent" />
                <span>Empirical Validation Cohort Reliability</span>
              </div>
              <p className="text-xs text-primary font-mono leading-relaxed bg-surface p-3 hairline-border rounded">
                &ldquo;{localReliability.statement}&rdquo;
              </p>
              <div className="flex items-center justify-between text-[11px] font-mono text-muted pt-1">
                <span>Validation Partition N = 14,902 (Bin sample: n = {localReliability.sample_size})</span>
                <span className="text-protective font-semibold">&bull; Reliability Bin Certified</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: GLOBAL MODEL KNOWLEDGE */}
      {activeTab === "global" && (
        <div className="p-4 sm:p-5 space-y-4">
          <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
            <div className="flex items-center justify-between border-b border-hairline pb-2.5">
              <h4 className="text-xs font-bold text-primary uppercase font-mono tracking-wider">
                Frozen Model Global Architecture & Multi-Method Concordance
              </h4>
              <span className="text-xs font-mono text-muted">
                {modelVersion}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <Stat
                label="Global Top Feature"
                value="Prior Inpatient"
                secondary="Mean |SHAP|: 0.2168"
                provenance={PROVENANCES.xaiTopFeatureShap}
              />
              <Stat
                label="Structured RAG Attribution"
                value="5.18% (Rank #7)"
                secondary="Qualitative Precedents"
                provenance={PROVENANCES.xaiRetrievalShare}
              />
              <Stat
                label="Method Agreement"
                value="rho = 0.953"
                secondary="TreeSHAP vs Gain"
                provenance={PROVENANCES.xaiMethodAgreementSpearman}
              />
            </div>

            <div className="p-3 bg-surface hairline-border rounded-sm space-y-2 text-xs font-mono">
              <div className="font-semibold text-primary">RAG Explanation vs Discrimination Lift:</div>
              <p className="text-muted font-sans text-xs leading-relaxed">
                The structured retrieval feature (rag_struct_rate_k20) ranks #7 globally, carrying 5.2% share of model attribution. This connects directly to ablation findings: while RAG provides rich qualitative precedent evidence for clinicians, it does not provide a statistically significant AUC boost (+0.0009 AUC, 95% CI [-0.0016, 0.0036]), confirming qualitative precedent value without statistical over-claiming.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
