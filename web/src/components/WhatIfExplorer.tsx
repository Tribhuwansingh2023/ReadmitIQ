import React, { useState, useEffect } from "react";
import {
  Sliders,
  RotateCcw,
  TrendingDown,
  TrendingUp,
  Info,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { PatientFormInput, WhatIfResponse } from "../types";

interface WhatIfExplorerProps {
  dischargeId: number;
  initialFeatures: PatientFormInput;
  baselineRawScore: number;
  baselineCalibratedProb: number;
  baselineTier: string;
}

export const WhatIfExplorer: React.FC<WhatIfExplorerProps> = ({
  dischargeId,
  initialFeatures,
  baselineRawScore,
  baselineCalibratedProb,
  baselineTier,
}) => {
  // Configurable features state
  const [inputs, setInputs] = useState<{
    time_in_hospital: number;
    number_inpatient: number;
    num_medications: number;
    number_emergency: number;
    num_lab_procedures: number;
    number_diagnoses: number;
    A1Cresult: string;
    insulin: string;
    change: string;
  }>({
    time_in_hospital: Number(initialFeatures.time_in_hospital) || 4,
    number_inpatient: Number(initialFeatures.number_inpatient) || 0,
    num_medications: Number(initialFeatures.num_medications) || 12,
    number_emergency: Number(initialFeatures.number_emergency) || 0,
    num_lab_procedures: Number(initialFeatures.num_lab_procedures) || 40,
    number_diagnoses: Number(initialFeatures.number_diagnoses) || 7,
    A1Cresult: String(initialFeatures.A1Cresult || "None"),
    insulin: String(initialFeatures.insulin || "No"),
    change: String(initialFeatures.change || "No"),
  });

  const [simulation, setSimulation] = useState<WhatIfResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Run simulation against backend
  const runSimulation = async (modifiedVals = inputs) => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.whatIfSensitivity({
        discharge_id: dischargeId,
        modified_features: modifiedVals,
      });
      setSimulation(res);
    } catch (err: any) {
      setError(err.message || "Failed to calculate what-if sensitivity");
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    runSimulation();
  }, [dischargeId]);

  // Debounced auto-recalculate on feature adjustment
  useEffect(() => {
    const timer = setTimeout(() => {
      runSimulation(inputs);
    }, 350);
    return () => clearTimeout(timer);
  }, [inputs]);

  const handleReset = () => {
    const original = {
      time_in_hospital: Number(initialFeatures.time_in_hospital) || 4,
      number_inpatient: Number(initialFeatures.number_inpatient) || 0,
      num_medications: Number(initialFeatures.num_medications) || 12,
      number_emergency: Number(initialFeatures.number_emergency) || 0,
      num_lab_procedures: Number(initialFeatures.num_lab_procedures) || 40,
      number_diagnoses: Number(initialFeatures.number_diagnoses) || 7,
      A1Cresult: String(initialFeatures.A1Cresult || "None"),
      insulin: String(initialFeatures.insulin || "No"),
      change: String(initialFeatures.change || "No"),
    };
    setInputs(original);
    runSimulation(original);
  };

  const delta = simulation?.delta;
  const modCalibrated = simulation?.modified?.calibrated_prob ?? baselineCalibratedProb;
  const modRaw = simulation?.modified?.raw_score ?? baselineRawScore;
  const modTier = simulation?.modified?.tier ?? (simulation?.modified as any)?.risk_tier ?? baselineTier;
  const probChange = delta ? (delta.probability_change ?? delta.calibrated_prob) : 0;
  const isReduced = probChange < 0;
  const isIncreased = probChange > 0;

  return (
    <div className="space-y-4 font-sans">
      {/* Header & Scientific Disclaimer */}
      <div className="bg-canvas hairline-border rounded-sm p-4 space-y-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-hairline pb-2">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] uppercase px-1.5 py-0.5 bg-accent/10 text-accent font-bold rounded">
              Differentiator 1
            </span>
            <h3 className="text-sm font-semibold text-primary flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-accent" />
              <span>What-If Risk Sensitivity Explorer</span>
            </h3>
          </div>

          <button
            onClick={handleReset}
            className="px-2.5 py-1 bg-surface hairline-border hover:bg-subtle text-muted hover:text-primary rounded text-xs font-mono flex items-center gap-1"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset to Baseline</span>
          </button>
        </div>

        {/* Mandatory Methodological Guardrail */}
        <div className="p-2.5 bg-subtle/80 hairline-border rounded text-[11px] text-muted space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-primary">
            <Info className="w-3.5 h-3.5 text-accent shrink-0" />
            <span>Sensitivity Analysis vs Causal Inference</span>
          </div>
          <p className="leading-relaxed">
            {simulation?.disclaimer ||
              "Model Sensitivity Analysis: Simulates frozen tabular model response to hypothetical input variations. This is NOT a causal estimate, clinical treatment effect, or guarantee of reduced readmission risk."}
          </p>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-critical/10 border border-critical/30 rounded text-xs font-mono text-critical">
          {error}
        </div>
      )}

      {/* Side-by-Side Comparison KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Baseline Card */}
        <div className="p-3 bg-surface hairline-border rounded-sm space-y-1 text-xs">
          <span className="font-mono text-[10px] uppercase text-muted block">
            Baseline Scored
          </span>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-2xl font-bold text-primary">
              {(baselineCalibratedProb * 100).toFixed(1)}%
            </span>
            <span className="font-mono text-[11px] px-1.5 py-0.2 bg-subtle text-muted rounded uppercase font-semibold">
              {baselineTier}
            </span>
          </div>
          <div className="text-[10px] font-mono text-muted pt-1 border-t border-hairline/60">
            Raw score: {baselineRawScore.toFixed(4)}
          </div>
        </div>

        {/* Modified Simulated Card */}
        <div className="p-3 bg-surface hairline-border rounded-sm space-y-1 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase text-muted block">
              Simulated Modified Risk
            </span>
            {loading && (
              <span className="font-mono text-[10px] text-accent animate-pulse">
                Scoring...
              </span>
            )}
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-2xl font-bold text-accent">
              {(modCalibrated * 100).toFixed(1)}%
            </span>
            <span className="font-mono text-[11px] px-1.5 py-0.2 bg-accent/10 text-accent rounded uppercase font-semibold">
              {modTier}
            </span>
          </div>
          <div className="text-[10px] font-mono text-muted pt-1 border-t border-hairline/60">
            Raw score: {modRaw.toFixed(4)}
          </div>
        </div>

        {/* Delta Card */}
        <div className="p-3 bg-surface hairline-border rounded-sm space-y-1 text-xs">
          <span className="font-mono text-[10px] uppercase text-muted block">
            Absolute Sensitivity Shift
          </span>
          <div className="flex items-center gap-2">
            {isReduced ? (
              <div className="flex items-center gap-1 text-protective font-mono text-xl font-bold">
                <TrendingDown className="w-5 h-5" />
                <span>{(probChange * 100).toFixed(1)}%</span>
              </div>
            ) : isIncreased ? (
              <div className="flex items-center gap-1 text-critical font-mono text-xl font-bold">
                <TrendingUp className="w-5 h-5" />
                <span>+{(probChange * 100).toFixed(1)}%</span>
              </div>
            ) : (
              <span className="font-mono text-xl font-bold text-muted">
                0.0% (Unchanged)
              </span>
            )}
          </div>
          <div className="text-[10px] font-mono text-muted pt-1 border-t border-hairline/60">
            {delta?.relative_risk_change_pct !== undefined
              ? `${delta.relative_risk_change_pct > 0 ? "+" : ""}${delta.relative_risk_change_pct}% relative change`
              : `${((probChange / (baselineCalibratedProb || 1)) * 100).toFixed(1)}% relative change`}
          </div>
        </div>
      </div>

      {/* Interactive Feature Controls */}
      <div className="p-4 bg-surface hairline-border rounded-sm space-y-4">
        <div className="border-b border-hairline pb-2 flex items-center justify-between">
          <h4 className="text-xs font-semibold text-primary uppercase font-mono tracking-wider">
            Supported Modifiable Encounter Inputs
          </h4>
          <span className="text-[10px] text-muted font-mono">
            Adjust sliders to evaluate model responsiveness
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 text-xs">
          {/* Length of stay */}
          <div className="space-y-1">
            <div className="flex justify-between font-mono text-[11px]">
              <span className="text-muted">Length of Stay (Days):</span>
              <span className="text-primary font-bold">
                {inputs.time_in_hospital}d
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="14"
              value={inputs.time_in_hospital}
              onChange={(e) =>
                setInputs({
                  ...inputs,
                  time_in_hospital: parseInt(e.target.value) || 1,
                })
              }
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[9px] font-mono text-muted">
              <span>1 day</span>
              <span>14 days</span>
            </div>
          </div>

          {/* Prior Inpatient Admissions */}
          <div className="space-y-1">
            <div className="flex justify-between font-mono text-[11px]">
              <span className="text-muted">Prior Inpatient Admissions:</span>
              <span className="text-primary font-bold">
                {inputs.number_inpatient}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="10"
              value={inputs.number_inpatient}
              onChange={(e) =>
                setInputs({
                  ...inputs,
                  number_inpatient: parseInt(e.target.value) || 0,
                })
              }
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[9px] font-mono text-muted">
              <span>0 visits</span>
              <span>10 visits</span>
            </div>
          </div>

          {/* Active Medications */}
          <div className="space-y-1">
            <div className="flex justify-between font-mono text-[11px]">
              <span className="text-muted">Total Medications:</span>
              <span className="text-primary font-bold">
                {inputs.num_medications}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="40"
              value={inputs.num_medications}
              onChange={(e) =>
                setInputs({
                  ...inputs,
                  num_medications: parseInt(e.target.value) || 1,
                })
              }
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[9px] font-mono text-muted">
              <span>1 med</span>
              <span>40 meds</span>
            </div>
          </div>

          {/* Prior Emergency Encounters */}
          <div className="space-y-1">
            <div className="flex justify-between font-mono text-[11px]">
              <span className="text-muted">Emergency Encounters:</span>
              <span className="text-primary font-bold">
                {inputs.number_emergency}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="8"
              value={inputs.number_emergency}
              onChange={(e) =>
                setInputs({
                  ...inputs,
                  number_emergency: parseInt(e.target.value) || 0,
                })
              }
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[9px] font-mono text-muted">
              <span>0 ER</span>
              <span>8 ER</span>
            </div>
          </div>

          {/* Number of Lab Procedures */}
          <div className="space-y-1">
            <div className="flex justify-between font-mono text-[11px]">
              <span className="text-muted">Lab Procedures:</span>
              <span className="text-primary font-bold">
                {inputs.num_lab_procedures}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              value={inputs.num_lab_procedures}
              onChange={(e) =>
                setInputs({
                  ...inputs,
                  num_lab_procedures: parseInt(e.target.value) || 1,
                })
              }
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[9px] font-mono text-muted">
              <span>1 lab</span>
              <span>100 labs</span>
            </div>
          </div>

          {/* Number of Diagnoses */}
          <div className="space-y-1">
            <div className="flex justify-between font-mono text-[11px]">
              <span className="text-muted">Diagnoses Count:</span>
              <span className="text-primary font-bold">
                {inputs.number_diagnoses}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="16"
              value={inputs.number_diagnoses}
              onChange={(e) =>
                setInputs({
                  ...inputs,
                  number_diagnoses: parseInt(e.target.value) || 1,
                })
              }
              className="w-full accent-accent cursor-pointer"
            />
            <div className="flex justify-between text-[9px] font-mono text-muted">
              <span>1</span>
              <span>16</span>
            </div>
          </div>
        </div>

        {/* Categorical Regimen Dropdowns */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 border-t border-hairline text-xs font-mono">
          <div>
            <label className="block text-muted text-[11px] mb-1">
              HbA1c Lab Result:
            </label>
            <select
              value={inputs.A1Cresult}
              onChange={(e) =>
                setInputs({ ...inputs, A1Cresult: e.target.value })
              }
              className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary text-xs"
            >
              <option value="None">None (Unmeasured)</option>
              <option value="Norm">Norm (Normal)</option>
              <option value=">7">&gt;7 (Elevated)</option>
              <option value=">8">&gt;8 (Severely Elevated)</option>
            </select>
          </div>

          <div>
            <label className="block text-muted text-[11px] mb-1">
              Insulin Regimen:
            </label>
            <select
              value={inputs.insulin}
              onChange={(e) => setInputs({ ...inputs, insulin: e.target.value })}
              className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary text-xs"
            >
              <option value="No">No (None)</option>
              <option value="Steady">Steady</option>
              <option value="Up">Up (Titrated)</option>
              <option value="Down">Down (Reduced)</option>
            </select>
          </div>

          <div>
            <label className="block text-muted text-[11px] mb-1">
              Diabetes Med Changed:
            </label>
            <select
              value={inputs.change}
              onChange={(e) => setInputs({ ...inputs, change: e.target.value })}
              className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary text-xs"
            >
              <option value="No">No (Unchanged)</option>
              <option value="Ch">Ch (Changed During Stay)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Section: SHAP Attribution Shifts & Driver Dynamics */}
      {simulation && (
        <div className="p-4 bg-surface hairline-border rounded-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 border-b border-hairline pb-2">
            <div>
              <h4 className="text-xs font-semibold text-primary uppercase font-mono tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-accent" />
                <span>Feature Attribution Shifts (Δ TreeSHAP Log-Odds)</span>
              </h4>
              <p className="text-[11px] text-muted font-mono">
                Evaluates how input modifications alter feature contributions in frozen tree paths
              </p>
            </div>
            {simulation.delta?.raw_margin !== undefined && (
              <span className="text-[10px] font-mono text-muted">
                Margin Shift: Δm = {simulation.delta.raw_margin >= 0 ? `+${simulation.delta.raw_margin.toFixed(4)}` : simulation.delta.raw_margin.toFixed(4)}
              </span>
            )}
          </div>

          {/* Newly Appearing or Disappearing Driver Badges */}
          {((simulation.newly_appearing_drivers && simulation.newly_appearing_drivers.length > 0) ||
            (simulation.disappearing_drivers && simulation.disappearing_drivers.length > 0)) && (
            <div className="p-3 bg-canvas hairline-border rounded-sm space-y-2 text-xs font-mono">
              <div className="text-[10px] uppercase font-bold text-muted">
                Top Risk Driver Transitions
              </div>
              <div className="flex flex-wrap gap-2">
                {simulation.newly_appearing_drivers?.map((feat, idx) => (
                  <span
                    key={`new-${idx}`}
                    className="px-2 py-0.5 rounded bg-critical/15 text-critical text-[11px] font-medium flex items-center gap-1"
                  >
                    <span>+ New Top Driver:</span>
                    <strong>{feat}</strong>
                  </span>
                ))}
                {simulation.disappearing_drivers?.map((feat, idx) => (
                  <span
                    key={`drop-${idx}`}
                    className="px-2 py-0.5 rounded bg-protective/15 text-protective text-[11px] font-medium flex items-center gap-1"
                  >
                    <span>- Displaced Driver:</span>
                    <strong>{feat}</strong>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Attribution Shifts Table */}
          {simulation.attribution_shifts && simulation.attribution_shifts.length > 0 ? (
            <div className="border border-hairline rounded overflow-hidden">
              <table className="w-full text-left text-xs font-sans border-collapse">
                <thead className="bg-subtle text-[10px] font-mono text-muted uppercase border-b border-hairline">
                  <tr>
                    <th className="p-2.5">Feature</th>
                    <th className="p-2.5 text-right">Baseline SHAP</th>
                    <th className="p-2.5 text-right">Simulated SHAP</th>
                    <th className="p-2.5 text-right">Shift (Δ φᵢ)</th>
                    <th className="p-2.5">Sensitivity Effect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hairline font-mono text-[11px]">
                  {simulation.attribution_shifts.map((shift, idx) => {
                    const isIncreased = shift.delta_shap > 0.0001;
                    const isReduced = shift.delta_shap < -0.0001;

                    return (
                      <tr key={idx} className="hover:bg-subtle/40">
                        <td className="p-2.5 text-primary">
                          <div className="font-semibold">{shift.feature_name}</div>
                          <div className="text-[10px] text-muted flex items-center gap-1">
                            <span>{shift.feature}</span>
                            {shift.is_modified_input && (
                              <span className="px-1 py-0.2 rounded bg-accent/15 text-accent text-[9px] font-bold">
                                Modified Input
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="p-2.5 text-right text-muted">
                          {shift.baseline_shap >= 0 ? `+${shift.baseline_shap.toFixed(4)}` : shift.baseline_shap.toFixed(4)}
                        </td>
                        <td className="p-2.5 text-right text-primary font-medium">
                          {shift.modified_shap >= 0 ? `+${shift.modified_shap.toFixed(4)}` : shift.modified_shap.toFixed(4)}
                        </td>
                        <td
                          className={`p-2.5 text-right font-bold ${
                            isIncreased
                              ? "text-critical"
                              : isReduced
                              ? "text-protective"
                              : "text-muted"
                          }`}
                        >
                          {shift.delta_shap > 0 ? `+${shift.delta_shap.toFixed(4)}` : shift.delta_shap.toFixed(4)}
                        </td>
                        <td className="p-2.5">
                          {isIncreased ? (
                            <span className="text-critical text-[10px] font-semibold">
                              Increased Risk Push
                            </span>
                          ) : isReduced ? (
                            <span className="text-protective text-[10px] font-semibold">
                              Reduced Risk Push
                            </span>
                          ) : (
                            <span className="text-muted text-[10px]">Unchanged</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-4 text-center text-xs text-muted font-mono">
              Adjust sliders above to observe feature attribution shifts.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
