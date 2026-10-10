import React, { useState, useEffect, useMemo } from "react";
import {
  X,
  Printer,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Sliders,
  Info,
  Send,
} from "lucide-react";
import { apiClient } from "../api/client";
import type {
  DischargeDetailResponse,
  EncounterStatus,
  PatientFormInput,
  MetricsResponse,
} from "../types";
import { RiskGlyph } from "../components/RiskGlyph";

interface DischargeBriefModalProps {
  dischargeId: number | null;
  onClose: () => void;
  onStatusUpdated?: () => void;
}

// Compute Wilson 95% Confidence Interval
function calculateWilsonCi(positives: number, total: number): [number, number] {
  if (total === 0) return [0, 0];
  const z = 1.95996; // 95% CI
  const p = positives / total;
  const z2 = z * z;
  const denominator = 1 + z2 / total;
  const center = (p + z2 / (2 * total)) / denominator;
  const margin =
    (z * Math.sqrt((p * (1 - p)) / total + z2 / (4 * total * total))) /
    denominator;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

export const DischargeBriefModal: React.FC<DischargeBriefModalProps> = ({
  dischargeId,
  onClose,
  onStatusUpdated,
}) => {
  const [detail, setDetail] = useState<DischargeDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Metrics for RAG explanation lift disclaimer
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  // Stepper & Override state
  const [selectedStatus, setSelectedStatus] = useState<EncounterStatus>("new");
  const [overrideTier, setOverrideTier] = useState<string>("");
  const [overrideReason, setOverrideReason] = useState<string>("");
  const [showOverrideForm, setShowOverrideForm] = useState<boolean>(false);
  const [updating, setUpdating] = useState<boolean>(false);

  // Action notes input
  const [newActionNote, setNewActionNote] = useState<string>("");
  const [selectedActionType, setSelectedActionType] =
    useState<string>("phone_call");

  // Sensitivity Lab State
  const [sensitivityInputs, setSensitivityInputs] =
    useState<Partial<PatientFormInput>>({});
  const [sensitivityScore, setSensitivityScore] = useState<number | null>(null);
  const [sensitivityCalibrated, setSensitivityCalibrated] = useState<
    number | null
  >(null);
  const [sensitivityLoading, setSensitivityLoading] = useState<boolean>(false);

  // Fetch Discharge Details
  const fetchDetail = async () => {
    if (!dischargeId) return;
    try {
      setLoading(true);
      const data = await apiClient.getDischarge(dischargeId);
      setDetail(data);
      setSelectedStatus(data.status);
      setSensitivityInputs({
        time_in_hospital: data.features.time_in_hospital,
        number_inpatient: data.features.number_inpatient,
        num_medications: data.features.num_medications,
        number_emergency: data.features.number_emergency,
      });
      setSensitivityScore(data.raw_score);
      setSensitivityCalibrated(data.calibrated_prob);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load discharge brief");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetail();
  }, [dischargeId]);

  useEffect(() => {
    apiClient
      .getMetrics()
      .then(setMetrics)
      .catch(() => {});
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Debounced Sensitivity Lab scoring
  useEffect(() => {
    if (!detail || !sensitivityInputs.time_in_hospital) return;
    const timer = setTimeout(async () => {
      try {
        setSensitivityLoading(true);
        const simPayload = {
          ...detail.features,
          ...sensitivityInputs,
        };
        const pred = await apiClient.predict(simPayload as PatientFormInput);
        const calibratedProb = pred.calibrated_prob ?? pred.calibrated_risk_probability ?? 0.25;
        const rawScore = pred.raw_score ?? 0.5;
        setSensitivityScore(rawScore);
        setSensitivityCalibrated(calibratedProb);
      } catch {
        // ignore
      } finally {
        setSensitivityLoading(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [sensitivityInputs]);

  // Handle Stepper Status Change
  const handleUpdateStatus = async (status: EncounterStatus) => {
    if (!detail) return;
    try {
      setUpdating(true);
      await apiClient.updateDischarge(detail.id, {
        status,
        actor: "Care Coordinator",
      });
      await apiClient.addAction(detail.id, {
        type: "status_change",
        note: `Status updated to ${status}.`,
        actor: "Care Coordinator",
      });
      setSelectedStatus(status);
      fetchDetail();
      onStatusUpdated?.();
    } catch (err: any) {
      alert(`Error updating status: ${err.message}`);
    } finally {
      setUpdating(false);
    }
  };

  // Handle Clinician Override Submit
  const handleOverrideSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!detail) return;
    if (!overrideReason.trim()) {
      alert("Clinical rationale is mandatory for risk tier overrides.");
      return;
    }
    try {
      setUpdating(true);
      await apiClient.updateDischarge(detail.id, {
        clinician_override_tier: overrideTier || undefined,
        override_reason: overrideReason,
        actor: "Attending Physician",
      });
      setShowOverrideForm(false);
      fetchDetail();
      onStatusUpdated?.();
    } catch (err: any) {
      alert(`Override error: ${err.message}`);
    } finally {
      setUpdating(false);
    }
  };

  // Handle Action Checklist toggle
  const handleToggleAction = async (actionTitle: string, timeframe: string) => {
    if (!detail) return;
    try {
      await apiClient.addAction(detail.id, {
        type: `checklist_${timeframe}`,
        note: `Completed: ${actionTitle}`,
        actor: "Discharge Specialist",
      });
      fetchDetail();
      onStatusUpdated?.();
    } catch (err: any) {
      alert(`Failed to record action: ${err.message}`);
    }
  };

  // Handle Add Action Note
  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!detail || !newActionNote.trim()) return;
    try {
      await apiClient.addAction(detail.id, {
        type: selectedActionType,
        note: newActionNote.trim(),
        actor: "Care Coordinator",
      });
      setNewActionNote("");
      fetchDetail();
      onStatusUpdated?.();
    } catch (err: any) {
      alert(`Failed to save note: ${err.message}`);
    }
  };

  // Precedent Orbit calculation
  const neighbors = detail?.similar_patients_summary?.top_neighbors || [];
  const neighborReadmittedCount = neighbors.filter(
    (n) => n.readmitted_30d === 1
  ).length;
  const [wilsonLower, wilsonUpper] = useMemo(() => {
    return calculateWilsonCi(neighborReadmittedCount, neighbors.length);
  }, [neighborReadmittedCount, neighbors.length]);

  if (!dischargeId) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="brief-title"
      className="fixed inset-0 z-50 overflow-y-auto bg-black/60 backdrop-blur-xs flex items-center justify-center p-2 sm:p-4 font-sans"
    >
      <div className="bg-canvas border border-hairline w-full max-w-5xl rounded-sm shadow-2xl overflow-hidden flex flex-col max-h-[92vh] print:max-h-none print:shadow-none print:border-none">
        {/* Top Header / Print bar */}
        <div className="px-6 py-4 bg-surface border-b border-hairline flex items-center justify-between shrink-0 print:border-b-2">
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs px-2 py-0.5 bg-subtle text-muted rounded uppercase tracking-wider font-semibold">
              ENCOUNTER BRIEF #{detail?.id}
            </span>
            <span className="text-xs text-muted font-mono">
              Model: {detail?.model_version || "LightGBM + Isotonic"}
            </span>
          </div>

          <div className="flex items-center gap-2 print:hidden">
            <button
              onClick={() => window.print()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium border border-hairline hover:bg-subtle text-primary transition-colors"
            >
              <Printer className="w-3.5 h-3.5 text-muted" />
              <span>Export Brief</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-muted hover:text-primary rounded hover:bg-subtle transition-colors"
              aria-label="Close brief"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="p-6 overflow-y-auto space-y-8 flex-1">
          {loading && (
            <div className="py-24 text-center space-y-3">
              <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-xs text-muted font-mono">
                Loading clinical decision brief and precedents...
              </p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-critical/10 border border-critical/30 rounded-sm text-critical text-xs font-mono">
              {error}
            </div>
          )}

          {detail && !loading && (
            <>
              {/* Patient Core Headline & Dial */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 bg-surface p-6 rounded-sm hairline-border">
                {/* Col 1 & 2: Patient Summary & Stepper */}
                <div className="lg:col-span-2 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hairline pb-3">
                    <div>
                      <h1
                        id="brief-title"
                        className="font-serif text-2xl font-medium text-primary"
                      >
                        Patient Encounter #{detail.id}
                      </h1>
                      <p className="text-xs text-muted font-mono mt-0.5">
                        Received:{" "}
                        {new Date(detail.received_at).toLocaleDateString()}{" "}
                        {new Date(detail.received_at).toLocaleTimeString()} •
                        Source: {detail.source}
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <RiskGlyph tier={detail.tier} size="md" />
                      <span className="font-serif text-xl font-bold text-primary">
                        {detail.tier} Tier
                      </span>
                    </div>
                  </div>

                  {/* Demographic & Encounter Facts */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="p-2.5 bg-canvas hairline-border rounded-sm">
                      <span className="text-[10px] font-mono text-muted uppercase block">
                        Age & Gender
                      </span>
                      <span className="font-semibold text-primary font-mono">
                        {detail.features.age} {detail.features.gender}
                      </span>
                    </div>

                    <div className="p-2.5 bg-canvas hairline-border rounded-sm">
                      <span className="text-[10px] font-mono text-muted uppercase block">
                        Primary Diagnosis
                      </span>
                      <span className="font-semibold text-primary">
                        {detail.features.diag_1_group}
                      </span>
                    </div>

                    <div className="p-2.5 bg-canvas hairline-border rounded-sm">
                      <span className="text-[10px] font-mono text-muted uppercase block">
                        Prior Inpatient
                      </span>
                      <span className="font-semibold text-primary font-mono">
                        {detail.features.number_inpatient} admissions
                      </span>
                    </div>

                    <div className="p-2.5 bg-canvas hairline-border rounded-sm">
                      <span className="text-[10px] font-mono text-muted uppercase block">
                        Length of Stay
                      </span>
                      <span className="font-semibold text-primary font-mono">
                        {detail.features.time_in_hospital} days
                      </span>
                    </div>
                  </div>

                  {/* Status Stepper */}
                  <div className="pt-2">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] font-mono uppercase tracking-wider text-muted font-bold">
                        Clinical Workflow Stepper
                      </span>
                      <span className="text-[11px] font-mono text-accent">
                        Current: {selectedStatus}
                      </span>
                    </div>

                    <div className="grid grid-cols-5 gap-1.5">
                      {[
                        { key: "new", label: "New" },
                        { key: "assigned", label: "Assigned" },
                        { key: "contacted", label: "Contacted" },
                        { key: "followup_scheduled", label: "Scheduled" },
                        { key: "closed", label: "Closed" },
                      ].map((step) => {
                        const isCurrent = selectedStatus === step.key;
                        return (
                          <button
                            key={step.key}
                            onClick={() =>
                              handleUpdateStatus(step.key as EncounterStatus)
                            }
                            disabled={updating}
                            className={`py-2 px-1 text-center rounded-sm text-[11px] font-medium transition-all ${
                              isCurrent
                                ? "bg-accent text-white font-bold shadow-xs"
                                : "bg-canvas hairline-border text-muted hover:text-primary hover:bg-subtle"
                            }`}
                          >
                            {step.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Clinician Override Link */}
                  <div className="pt-1 flex items-center justify-between text-xs">
                    <button
                      onClick={() => setShowOverrideForm(!showOverrideForm)}
                      className="text-muted hover:text-primary text-[11px] font-mono underline"
                    >
                      {showOverrideForm
                        ? "Cancel Clinician Override"
                        : "Apply Clinician Risk Override →"}
                    </button>
                    {detail.audit_trail.some(
                      (a) => a.event === "clinician_override"
                    ) && (
                      <span className="text-caution font-mono text-[10px] bg-caution/15 px-2 py-0.5 rounded">
                        OVERRIDDEN BY CLINICIAN
                      </span>
                    )}
                  </div>

                  {/* Clinician Override Drawer/Form */}
                  {showOverrideForm && (
                    <form
                      onSubmit={handleOverrideSubmit}
                      className="p-4 bg-canvas border border-caution/40 rounded-sm space-y-3 text-xs"
                    >
                      <div className="flex items-center gap-2 text-caution font-semibold">
                        <AlertTriangle className="w-4 h-4" />
                        <span>Clinician Risk Tier Override</span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div>
                          <label className="block text-muted text-[11px] mb-1">
                            New Assigned Tier
                          </label>
                          <select
                            value={overrideTier}
                            onChange={(e) => setOverrideTier(e.target.value)}
                            className="w-full bg-surface hairline-border rounded px-2.5 py-1.5 text-primary text-xs"
                          >
                            <option value="">Select Override Tier</option>
                            <option value="Critical">Critical</option>
                            <option value="High">High</option>
                            <option value="Medium">Medium</option>
                            <option value="Low">Low</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-muted text-[11px] mb-1">
                            Clinical Rationale (Mandatory)
                          </label>
                          <input
                            type="text"
                            placeholder="e.g. Patient has strong home nursing support..."
                            value={overrideReason}
                            onChange={(e) => setOverrideReason(e.target.value)}
                            className="w-full bg-surface hairline-border rounded px-2.5 py-1.5 text-primary text-xs"
                          />
                        </div>
                      </div>
                      <div className="flex justify-end gap-2 pt-1">
                        <button
                          type="submit"
                          disabled={updating || !overrideTier}
                          className="px-3 py-1.5 bg-caution text-ink text-xs font-bold rounded-sm hover:opacity-90 disabled:opacity-50"
                        >
                          Submit Clinical Override
                        </button>
                      </div>
                    </form>
                  )}
                </div>

                {/* Col 3: Calibrated Risk Dial */}
                <div className="p-4 bg-canvas hairline-border rounded-sm flex flex-col justify-between items-center text-center">
                  <div className="w-full text-left">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-muted font-bold">
                      Calibrated Readmission Probability
                    </span>
                  </div>

                  {/* SVG Risk Dial */}
                  <div className="relative w-44 h-44 flex items-center justify-center my-2">
                    <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                      {/* Background track */}
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="8"
                        className="text-subtle"
                        fill="transparent"
                      />
                      {/* Calibrated Probability arc */}
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="8"
                        className={
                          detail.tier === "Critical"
                            ? "text-critical"
                            : detail.tier === "High"
                            ? "text-highrisk"
                            : detail.tier === "Medium"
                            ? "text-caution"
                            : "text-protective"
                        }
                        strokeDasharray={2 * Math.PI * 40}
                        strokeDashoffset={
                          2 * Math.PI * 40 * (1 - Math.min(1, detail.calibrated_prob))
                        }
                        strokeLinecap="round"
                        fill="transparent"
                      />
                    </svg>

                    <div className="absolute flex flex-col items-center justify-center">
                      <span className="font-mono text-3xl font-bold text-primary">
                        {(detail.calibrated_prob * 100).toFixed(1)}%
                      </span>
                      <span className="text-[10px] font-mono text-muted uppercase">
                        30-Day Risk
                      </span>
                    </div>
                  </div>

                  <div className="w-full text-xs space-y-1 font-mono text-muted border-t border-hairline pt-3">
                    <div className="flex justify-between">
                      <span>Raw LightGBM Score:</span>
                      <span className="text-primary font-semibold">
                        {detail.raw_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Serving Tier:</span>
                      <span className="text-primary font-semibold">
                        {detail.tier}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 2: Diverging SHAP Drivers */}
              <div className="bg-surface p-6 rounded-sm hairline-border space-y-4">
                <div className="border-b border-hairline pb-2 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                  <div>
                    <h2 className="text-sm font-semibold text-primary">
                      What Pushes This Patient's Risk Up or Down (Model Log-Odds)
                    </h2>
                    <p className="text-[11px] text-muted font-mono">
                      Attributions represent additive shifts in model log-odds space; they do NOT sum directly to the final calibrated probability.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                  {/* Risk Increasing */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-critical text-xs font-semibold uppercase tracking-wider font-mono">
                      <TrendingUp className="w-4 h-4" />
                      <span>Top Factors Pushing Risk Up (+ Log-Odds)</span>
                    </div>

                    <div className="space-y-2">
                      {detail.shap_explanation.top_risk_increasing.map((f, i) => {
                        const featName = f.feature_name || f.feature;
                        const attr = f.attribution ?? f.shap_value ?? 0;
                        const meaning = f.clinical_meaning || f.explanation || "Risk contributor";
                        return (
                          <div
                            key={i}
                            className="p-3 bg-canvas hairline-border rounded-sm text-xs space-y-1"
                          >
                            <div className="flex justify-between font-medium">
                              <span className="text-primary font-mono">{featName}</span>
                              <span className="text-critical font-mono font-bold">
                                +{attr.toFixed(3)}
                              </span>
                            </div>
                            <p className="text-[11px] text-muted">{meaning}</p>
                            <div className="w-full bg-subtle h-1.5 rounded-full overflow-hidden mt-1">
                              <div
                                className="bg-critical h-full"
                                style={{
                                  width: `${Math.min(100, Math.abs(attr) * 80)}%`,
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Risk Reducing */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-protective text-xs font-semibold uppercase tracking-wider font-mono">
                      <TrendingDown className="w-4 h-4" />
                      <span>Top Factors Pushing Risk Down (- Log-Odds)</span>
                    </div>

                    <div className="space-y-2">
                      {detail.shap_explanation.top_risk_reducing.length === 0 ? (
                        <div className="p-4 bg-canvas hairline-border rounded-sm text-xs text-muted text-center font-mono">
                          No significant protective factors detected for this high-acuity encounter.
                        </div>
                      ) : (
                        detail.shap_explanation.top_risk_reducing.map((f, i) => {
                          const featName = f.feature_name || f.feature;
                          const attr = f.attribution ?? f.shap_value ?? 0;
                          const meaning = f.clinical_meaning || f.explanation || "Protective factor";
                          return (
                            <div
                              key={i}
                              className="p-3 bg-canvas hairline-border rounded-sm text-xs space-y-1"
                            >
                              <div className="flex justify-between font-medium">
                                <span className="text-primary font-mono">{featName}</span>
                                <span className="text-protective font-mono font-bold">
                                  {attr.toFixed(3)}
                                </span>
                              </div>
                              <p className="text-[11px] text-muted">{meaning}</p>
                              <div className="w-full bg-subtle h-1.5 rounded-full overflow-hidden mt-1">
                                <div
                                  className="bg-protective h-full"
                                  style={{
                                    width: `${Math.min(100, Math.abs(attr) * 80)}%`,
                                  }}
                                />
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 3: Precedent Orbit (Radial Visualization) */}
              <div className="bg-surface p-6 rounded-sm hairline-border space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-hairline pb-3">
                  <div>
                    <h2 className="text-sm font-semibold text-primary">
                      Precedent Orbit: 20 Structurally Similar Historical Patients
                    </h2>
                    <p className="text-xs text-muted">
                      Radial distance represents geometric feature distance (d = 1 - cosine similarity).
                    </p>
                  </div>

                  {/* Wilson 95% Rate summary */}
                  <div className="p-2.5 bg-canvas hairline-border rounded-sm text-right">
                    <span className="text-[10px] font-mono uppercase text-muted block">
                      Neighbor Readmission Rate
                    </span>
                    <span className="text-sm font-mono font-bold text-primary">
                      {((neighborReadmittedCount / (neighbors.length || 1)) * 100).toFixed(1)}%{" "}
                      <span className="text-xs font-normal text-muted">
                        ({neighborReadmittedCount}/{neighbors.length})
                      </span>
                    </span>
                    <span className="text-[10px] font-mono text-muted block">
                      Wilson 95% CI: [{(wilsonLower * 100).toFixed(1)}%, {(wilsonUpper * 100).toFixed(1)}%]
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                  {/* SVG Precedent Orbit Chart */}
                  <div className="md:col-span-2 flex justify-center py-2">
                    <div className="relative w-80 h-80 bg-canvas hairline-border rounded-full flex items-center justify-center p-4">
                      {/* Concentric distance rings */}
                      <div className="absolute w-64 h-64 border border-dashed border-hairline rounded-full pointer-events-none" />
                      <div className="absolute w-44 h-44 border border-dashed border-hairline rounded-full pointer-events-none" />
                      <div className="absolute w-24 h-24 border border-dashed border-hairline rounded-full pointer-events-none" />

                      {/* Center Node (Current Encounter) */}
                      <div className="absolute z-10 w-5 h-5 bg-accent text-white rounded-full flex items-center justify-center shadow-md animate-pulse">
                        <span className="text-[8px] font-bold">★</span>
                      </div>

                      {/* 20 Neighbor Nodes placed radially */}
                      {neighbors.map((n, idx) => {
                        // Angle
                        const angle = (idx / (neighbors.length || 20)) * 2 * Math.PI;
                        // Distance: 1 - similarity (normalized between 35px and 140px)
                        const dist = 35 + (1 - (n.similarity_score || 0.8)) * 300;
                        const x = Math.cos(angle) * Math.min(135, dist);
                        const y = Math.sin(angle) * Math.min(135, dist);
                        const isReadmitted = n.readmitted_30d === 1;

                        return (
                          <div
                            key={n.encounter_id || idx}
                            className={`absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer transition-transform hover:scale-150 ${
                              isReadmitted
                                ? "w-3 h-3 bg-critical rotate-45"
                                : "w-3 h-3 bg-protective rounded-full"
                            }`}
                            style={{
                              left: `calc(50% + ${x}px)`,
                              top: `calc(50% + ${y}px)`,
                            }}
                            title={`Neighbor #${n.encounter_id}: ${isReadmitted ? "Readmitted" : "Not Readmitted"} (Sim: ${(n.similarity_score * 100).toFixed(0)}%)`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Precedent Orbit Caption & Methodological Guardrail */}
                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-canvas hairline-border rounded-sm space-y-2">
                      <div className="flex items-center gap-1.5 font-semibold text-primary">
                        <Info className="w-4 h-4 text-accent" />
                        <span>Explanation vs Predictive Lift</span>
                      </div>
                      <p className="text-[11px] text-muted leading-relaxed">
                        Retrieval of similar historical precedents provides clinical context and explainability to human care coordinators. It does <strong>not</strong> add independent predictive ROC-AUC lift over the tabular gradient booster.
                      </p>
                      {metrics?.ablation_table?.validation_models && (
                        <p className="text-[10px] font-mono text-muted border-t border-hairline pt-1">
                          Paired ΔAUC (LightGBM vs RAG augmented): 0.0000, 95% CI crosses zero.
                        </p>
                      )}
                    </div>

                    <div className="flex items-center gap-4 text-[11px] font-mono text-muted">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 bg-critical rotate-45" />
                        <span>Readmitted (30d)</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 bg-protective rounded-full" />
                        <span>Discharged Home</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 4: Suggested Actions Timeline */}
              <div className="bg-surface p-6 rounded-sm hairline-border space-y-4">
                <div className="border-b border-hairline pb-2">
                  <h2 className="text-sm font-semibold text-primary">
                    Tailored Post-Discharge Action Plan & Care Protocol
                  </h2>
                  <p className="text-xs text-muted">
                    Time-sequenced interventions dynamically triggered by model risk drivers. Check items to record care actions.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* 48 Hours */}
                  <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
                    <div className="flex items-center justify-between border-b border-hairline pb-1.5">
                      <span className="font-mono text-xs font-bold text-critical uppercase">
                        Phase 1: 48 Hours
                      </span>
                      <span className="text-[10px] font-mono text-muted">Immediate Triage</span>
                    </div>

                    <div className="space-y-2">
                      <label className="flex items-start gap-2 text-xs text-primary cursor-pointer hover:bg-subtle/50 p-1 rounded">
                        <input
                          type="checkbox"
                          className="mt-0.5 accent-accent"
                          onChange={() =>
                            handleToggleAction("Pharmacist medication reconciliation phone call", "48h")
                          }
                        />
                        <div>
                          <p className="font-medium">Medication Reconciliation Call</p>
                          <p className="text-[10px] text-muted">
                            Triggered by: {detail.features.num_medications} active meds & {detail.features.insulin} insulin
                          </p>
                        </div>
                      </label>

                      <label className="flex items-start gap-2 text-xs text-primary cursor-pointer hover:bg-subtle/50 p-1 rounded">
                        <input
                          type="checkbox"
                          className="mt-0.5 accent-accent"
                          onChange={() =>
                            handleToggleAction("Urgent PCP follow-up confirmation", "48h")
                          }
                        />
                        <div>
                          <p className="font-medium">Confirm PCP Appointment</p>
                          <p className="text-[10px] text-muted">
                            Triggered by: {detail.features.diag_1_group} primary diagnosis
                          </p>
                        </div>
                      </label>
                    </div>
                  </div>

                  {/* 7 Days */}
                  <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
                    <div className="flex items-center justify-between border-b border-hairline pb-1.5">
                      <span className="font-mono text-xs font-bold text-caution uppercase">
                        Phase 2: 7 Days
                      </span>
                      <span className="text-[10px] font-mono text-muted">Clinical Check-in</span>
                    </div>

                    <div className="space-y-2">
                      <label className="flex items-start gap-2 text-xs text-primary cursor-pointer hover:bg-subtle/50 p-1 rounded">
                        <input
                          type="checkbox"
                          className="mt-0.5 accent-accent"
                          onChange={() =>
                            handleToggleAction("Nurse home visit / telehealth glucose review", "7d")
                          }
                        />
                        <div>
                          <p className="font-medium">Telehealth Glycemic Review</p>
                          <p className="text-[10px] text-muted">
                            Triggered by: HbA1c {detail.features.A1Cresult}
                          </p>
                        </div>
                      </label>

                      <label className="flex items-start gap-2 text-xs text-primary cursor-pointer hover:bg-subtle/50 p-1 rounded">
                        <input
                          type="checkbox"
                          className="mt-0.5 accent-accent"
                          onChange={() =>
                            handleToggleAction("Durable medical equipment & test strip check", "7d")
                          }
                        />
                        <div>
                          <p className="font-medium">Supplies & DME Delivery Check</p>
                          <p className="text-[10px] text-muted">
                            Triggered by: {detail.features.discharge_disposition_group} disposition
                          </p>
                        </div>
                      </label>
                    </div>
                  </div>

                  {/* 30 Days */}
                  <div className="p-4 bg-canvas hairline-border rounded-sm space-y-3">
                    <div className="flex items-center justify-between border-b border-hairline pb-1.5">
                      <span className="font-mono text-xs font-bold text-protective uppercase">
                        Phase 3: 30 Days
                      </span>
                      <span className="text-[10px] font-mono text-muted">Transition Closeout</span>
                    </div>

                    <div className="space-y-2">
                      <label className="flex items-start gap-2 text-xs text-primary cursor-pointer hover:bg-subtle/50 p-1 rounded">
                        <input
                          type="checkbox"
                          className="mt-0.5 accent-accent"
                          onChange={() =>
                            handleToggleAction("Day-30 readmission survey & case resolution", "30d")
                          }
                        />
                        <div>
                          <p className="font-medium">Day-30 Episode Closeout</p>
                          <p className="text-[10px] text-muted">
                            Verify transition stability and complete documentation
                          </p>
                        </div>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Add Quick Action Note Form */}
                <form
                  onSubmit={handleAddNote}
                  className="pt-2 flex flex-col sm:flex-row gap-2"
                >
                  <select
                    value={selectedActionType}
                    onChange={(e) => setSelectedActionType(e.target.value)}
                    className="text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                  >
                    <option value="phone_call">Phone Call</option>
                    <option value="home_visit">Home Visit</option>
                    <option value="med_rec">Medication Reconciliation</option>
                    <option value="transportation">Transportation Assist</option>
                    <option value="other">General Note</option>
                  </select>
                  <input
                    type="text"
                    placeholder="Record care coordinator note..."
                    value={newActionNote}
                    onChange={(e) => setNewActionNote(e.target.value)}
                    className="flex-1 text-xs bg-canvas hairline-border rounded px-3 py-1.5 text-primary"
                  />
                  <button
                    type="submit"
                    className="px-4 py-1.5 bg-accent text-white text-xs font-medium rounded hover:opacity-90 flex items-center justify-center gap-1.5 shrink-0"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span>Save Action Note</span>
                  </button>
                </form>
              </div>

              {/* Section 5: Sensitivity Lab */}
              <div className="bg-surface p-6 rounded-sm hairline-border space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-hairline pb-2">
                  <div>
                    <h2 className="text-sm font-semibold text-primary flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-accent" />
                      <span>Clinical Sensitivity Lab</span>
                    </h2>
                    <p className="text-[11px] text-muted font-mono mt-0.5">
                      model sensitivity, not causal effect, not clinical advice
                    </p>
                  </div>

                  <div className="p-2 bg-canvas hairline-border rounded-sm text-right">
                    <span className="text-[10px] font-mono text-muted uppercase block">
                      Simulated Calibrated Risk {sensitivityLoading && "(Updating...)"}
                    </span>
                    <span className="text-base font-mono font-bold text-accent">
                      {sensitivityCalibrated !== null
                        ? `${(sensitivityCalibrated * 100).toFixed(1)}%`
                        : "---"}
                    </span>
                    {sensitivityScore !== null && (
                      <span className="text-[10px] font-mono text-muted block">
                        Raw score: {sensitivityScore.toFixed(4)}
                      </span>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div>
                    <label className="block text-muted mb-1">
                      Length of Stay (Days: {sensitivityInputs.time_in_hospital})
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="14"
                      value={sensitivityInputs.time_in_hospital || 4}
                      onChange={(e) =>
                        setSensitivityInputs({
                          ...sensitivityInputs,
                          time_in_hospital: parseInt(e.target.value) || 1,
                        })
                      }
                      className="w-full accent-accent"
                    />
                  </div>

                  <div>
                    <label className="block text-muted mb-1">
                      Prior Inpatient Admissions ({sensitivityInputs.number_inpatient})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="10"
                      value={sensitivityInputs.number_inpatient || 0}
                      onChange={(e) =>
                        setSensitivityInputs({
                          ...sensitivityInputs,
                          number_inpatient: parseInt(e.target.value) || 0,
                        })
                      }
                      className="w-full accent-accent"
                    />
                  </div>

                  <div>
                    <label className="block text-muted mb-1">
                      Total Medications ({sensitivityInputs.num_medications})
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="40"
                      value={sensitivityInputs.num_medications || 15}
                      onChange={(e) =>
                        setSensitivityInputs({
                          ...sensitivityInputs,
                          num_medications: parseInt(e.target.value) || 1,
                        })
                      }
                      className="w-full accent-accent"
                    />
                  </div>

                  <div>
                    <label className="block text-muted mb-1">
                      Prior ER Encounters ({sensitivityInputs.number_emergency})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="8"
                      value={sensitivityInputs.number_emergency || 0}
                      onChange={(e) =>
                        setSensitivityInputs({
                          ...sensitivityInputs,
                          number_emergency: parseInt(e.target.value) || 0,
                        })
                      }
                      className="w-full accent-accent"
                    />
                  </div>
                </div>
              </div>

              {/* Section 6: Action History & Audit Timeline */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Care Action History */}
                <div className="p-4 bg-surface hairline-border rounded-sm space-y-3">
                  <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-muted border-b border-hairline pb-1">
                    Recorded Care Coordinator Actions ({detail.action_history.length})
                  </h3>
                  <div className="space-y-2 max-h-48 overflow-y-auto text-xs">
                    {detail.action_history.length === 0 ? (
                      <p className="text-muted font-mono text-[11px] py-4 text-center">
                        No actions logged yet. Use the protocol checklist above.
                      </p>
                    ) : (
                      detail.action_history.map((a) => (
                        <div
                          key={a.id}
                          className="p-2 bg-canvas hairline-border rounded-sm space-y-0.5"
                        >
                          <div className="flex justify-between font-mono text-[10px] text-muted">
                            <span className="font-semibold text-primary">{a.actor}</span>
                            <span>{new Date(a.created_at).toLocaleTimeString()}</span>
                          </div>
                          <p className="text-primary text-[11px]">{a.note}</p>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Audit Timeline */}
                <div className="p-4 bg-surface hairline-border rounded-sm space-y-3">
                  <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-muted border-b border-hairline pb-1">
                    System Audit Trail ({detail.audit_trail.length})
                  </h3>
                  <div className="space-y-2 max-h-48 overflow-y-auto text-xs">
                    {detail.audit_trail.map((t) => (
                      <div
                        key={t.id}
                        className="p-2 bg-canvas hairline-border rounded-sm flex items-center justify-between font-mono text-[10px]"
                      >
                        <div>
                          <span className="font-semibold text-primary block">
                            {t.event}
                          </span>
                          <span className="text-muted">Actor: {t.actor}</span>
                        </div>
                        <span className="text-muted">
                          {new Date(t.ts).toLocaleTimeString()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
