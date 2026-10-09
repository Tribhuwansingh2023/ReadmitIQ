import React, { useEffect, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  Award,
  BarChart3,
  CheckCircle2,
  Filter,
  Loader2,
  TrendingUp,
  Users,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { CohortPatient, CohortResponse, PatientFormInput } from "../types";

interface CohortPageProps {
  onSelectPatient: (patient: Partial<PatientFormInput>) => void;
}

export const CohortPage: React.FC<CohortPageProps> = ({ onSelectPatient }) => {
  const [capacity, setCapacity] = useState<number>(0.1);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [cohortData, setCohortData] = useState<CohortResponse | null>(null);

  // Impact Estimate Simulation State (User-entered assumptions)
  const [outreachVolume, setOutreachVolume] = useState<number>(1490);
  const [costPerReadmission, setCostPerReadmission] = useState<number>(15200);
  const [reductionRate, setReductionRate] = useState<number>(18);
  const [costPerIntervention, setCostPerIntervention] = useState<number>(120);

  const fetchCohort = async (cap: number) => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getCohort(cap);
      setCohortData(data);
      setOutreachVolume(data.flagged_encounters_count);
    } catch (err: any) {
      setError(err?.message || "Failed to load cohort replay");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCohort(capacity);
  }, [capacity]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setCapacity(val);
  };

  const mapRowToPatientInput = (
    p: CohortPatient,
  ): Partial<PatientFormInput> => {
    return {
      age: p.age,
      gender: p.gender,
      time_in_hospital: p.time_in_hospital,
      number_inpatient: p.number_inpatient,
      diag_1_group: p.diag_1_group,
    };
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-sky-500/10 text-sky-400 border border-sky-500/20">
              Retrospective Replay
            </span>
            <span className="text-xs text-slate-400">
              Held-Out Test Split (N = 14,903 encounters)
            </span>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Cohort Command Center & Intervention Triage
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Simulate proactive care-team capacity allocation and measure true
            30-day readmissions captured.
          </p>
        </div>

        {/* Capacity Slider Card */}
        <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 flex flex-col gap-2 min-w-[280px]">
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-400 font-medium flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-sky-400" />
              Intervention Capacity
            </span>
            <span className="font-mono text-base font-bold text-sky-400">
              {(capacity * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.30"
            step="0.01"
            value={capacity}
            onChange={handleSliderChange}
            className="w-full accent-sky-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
          />
          <div className="flex justify-between text-[11px] text-slate-500">
            <span>1% (Intensive)</span>
            <span>10% (Standard)</span>
            <span>30% (Broad)</span>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* KPI Cards */}
      {cohortData && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-md">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span>Flagged for Care Triage</span>
              <Users className="w-4 h-4 text-sky-400" />
            </div>
            <div className="text-3xl font-extrabold text-white font-mono">
              {cohortData.flagged_encounters_count.toLocaleString()}
            </div>
            <div className="text-xs text-slate-400 mt-2">
              of {cohortData.total_test_encounters.toLocaleString()} test
              encounters
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-md">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span>True Readmissions Caught</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-3xl font-extrabold text-emerald-400 font-mono">
              {cohortData.true_readmissions_caught.toLocaleString()}
            </div>
            <div className="text-xs text-slate-400 mt-2">
              {(cohortData.recall * 100).toFixed(1)}% sensitivity (
              {cohortData.true_readmissions_caught} /{" "}
              {cohortData.total_true_readmissions_in_cohort})
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-md">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span>Precision @ Capacity</span>
              <Award className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-3xl font-extrabold text-white font-mono">
              {(cohortData.precision * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-slate-400 mt-2">
              Cutoff Score ≥ {cohortData.threshold_at_capacity.toFixed(4)}
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-md">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span>Lift vs Random Triage</span>
              <TrendingUp className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-3xl font-extrabold text-purple-400 font-mono">
              {cohortData.lift_vs_random.toFixed(2)}x
            </div>
            <div className="text-xs text-slate-400 mt-2">
              Base cohort prevalence: 11.37%
            </div>
          </div>
        </div>
      )}

      {/* Capacity Impact Visualizer Bar */}
      {cohortData && (
        <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/60">
          <div className="flex items-center justify-between mb-3 text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-1.5">
              <BarChart3 className="w-4 h-4 text-sky-400" />
              Cohort Capture Efficiency
            </span>
            <span className="text-slate-400">
              Intervention effort captures{" "}
              <strong className="text-white">
                {(cohortData.recall * 100).toFixed(1)}%
              </strong>{" "}
              of all readmissions at{" "}
              <strong className="text-white">
                {(capacity * 100).toFixed(0)}%
              </strong>{" "}
              workload
            </span>
          </div>
          <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex">
            <div
              className="bg-emerald-500 transition-all duration-300"
              style={{ width: `${Math.min(100, cohortData.recall * 100)}%` }}
              title={`Readmissions caught: ${(cohortData.recall * 100).toFixed(1)}%`}
            />
            <div
              className="bg-slate-700/60 transition-all duration-300"
              style={{
                width: `${100 - Math.min(100, cohortData.recall * 100)}%`,
              }}
            />
          </div>
          <div className="flex justify-between items-center text-[11px] text-slate-500 mt-2">
            <span>0% caught</span>
            <span className="text-emerald-400 font-medium">
              Caught: {cohortData.true_readmissions_caught} patients
            </span>
            <span>
              100% ({cohortData.total_true_readmissions_in_cohort} total
              readmissions)
            </span>
          </div>
        </div>
      )}

      {/* Interactive Impact Estimate Simulation Box (What-If Analysis) */}
      {cohortData && (
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-5">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                  Illustrative What-If Scenario
                </span>
                <span className="text-xs text-slate-400">
                  Leadership Capacity & Financial Modeling
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                Projected Clinical & Economic Impact Estimate
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Simulate transition-of-care program outcomes under
                user-specified assumptions.
              </p>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300 max-w-sm">
              <strong>Notice:</strong> Illustrative simulation based on entered
              assumptions, not an observed randomized trial result.
            </div>
          </div>

          {/* User-Configurable Assumptions Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
            {/* Outreach Volume */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold text-slate-300 block uppercase tracking-wider">
                Outreach Cohort (Patients)
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="10"
                  max="14903"
                  value={outreachVolume}
                  onChange={(e) =>
                    setOutreachVolume(
                      Math.max(1, parseInt(e.target.value) || 1),
                    )
                  }
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:border-sky-500"
                />
                <button
                  onClick={() =>
                    setOutreachVolume(cohortData.flagged_encounters_count)
                  }
                  title="Sync with Capacity Slider"
                  className="px-2 py-1.5 text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 whitespace-nowrap"
                >
                  Sync
                </button>
              </div>
              <span className="text-[10px] text-slate-500 block">
                Default: {cohortData.flagged_encounters_count.toLocaleString()}{" "}
                (at {(capacity * 100).toFixed(0)}% capacity)
              </span>
            </div>

            {/* Cost per Readmission */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold text-slate-300 block uppercase tracking-wider">
                Cost per Readmission ($)
              </label>
              <input
                type="number"
                step="500"
                min="5000"
                max="50000"
                value={costPerReadmission}
                onChange={(e) =>
                  setCostPerReadmission(
                    Math.max(1000, parseInt(e.target.value) || 15200),
                  )
                }
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:border-sky-500"
              />
              <span className="text-[10px] text-slate-500 block">
                Default assumption: $15,200 (CMS national average)
              </span>
            </div>

            {/* Relative Risk Reduction Rate */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="text-[11px] font-semibold text-slate-300 block uppercase tracking-wider">
                  Outreach Reduction Rate
                </label>
                <span className="font-mono text-xs font-bold text-emerald-400">
                  {reductionRate}%
                </span>
              </div>
              <input
                type="range"
                min="5"
                max="35"
                step="1"
                value={reductionRate}
                onChange={(e) =>
                  setReductionRate(parseInt(e.target.value) || 18)
                }
                className="w-full accent-emerald-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
              />
              <span className="text-[10px] text-slate-500 block">
                Default assumption: 18% (Project RED / BOOST range: 12%–24%)
              </span>
            </div>

            {/* Cost per Outreach Call */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold text-slate-300 block uppercase tracking-wider">
                Cost per Intervention ($)
              </label>
              <input
                type="number"
                step="10"
                min="20"
                max="1000"
                value={costPerIntervention}
                onChange={(e) =>
                  setCostPerIntervention(
                    Math.max(10, parseInt(e.target.value) || 120),
                  )
                }
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:border-sky-500"
              />
              <span className="text-[10px] text-slate-500 block">
                Default assumption: $120 (nurse call + pharmacist recon)
              </span>
            </div>
          </div>

          {/* Simulation Output Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                True Readmissions in Group
              </span>
              <span className="text-2xl font-bold font-mono text-white mt-1 block">
                {Math.round(
                  outreachVolume * cohortData.precision,
                ).toLocaleString()}
              </span>
              <span className="text-[10px] text-slate-500 mt-1 block">
                Based on {(cohortData.precision * 100).toFixed(1)}% model
                precision
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-emerald-500/30">
              <span className="text-[11px] text-emerald-400 uppercase tracking-wider block font-medium">
                Readmissions Averted
              </span>
              <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
                {Math.round(
                  outreachVolume * cohortData.precision * (reductionRate / 100),
                ).toLocaleString()}
              </span>
              <span className="text-[10px] text-slate-400 mt-1 block">
                At {reductionRate}% assumed intervention reduction
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block font-medium">
                Gross Inpatient Savings
              </span>
              <span className="text-2xl font-bold font-mono text-sky-400 mt-1 block">
                $
                {(
                  Math.round(
                    outreachVolume *
                      cohortData.precision *
                      (reductionRate / 100),
                  ) * costPerReadmission
                ).toLocaleString()}
              </span>
              <span className="text-[10px] text-slate-500 mt-1 block">
                Avoided hospitalization expense
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-emerald-500/40 bg-emerald-500/5">
              <span className="text-[11px] text-emerald-300 uppercase tracking-wider block font-semibold">
                Estimated Net Financial Value
              </span>
              <span className="text-2xl font-extrabold font-mono text-emerald-300 mt-1 block">
                $
                {(
                  Math.round(
                    outreachVolume *
                      cohortData.precision *
                      (reductionRate / 100),
                  ) *
                    costPerReadmission -
                  outreachVolume * costPerIntervention
                ).toLocaleString()}
              </span>
              <span className="text-[10px] text-emerald-400/80 mt-1 block">
                Net ROI:{" "}
                {(
                  (Math.round(
                    outreachVolume *
                      cohortData.precision *
                      (reductionRate / 100),
                  ) *
                    costPerReadmission) /
                  Math.max(1, outreachVolume * costPerIntervention)
                ).toFixed(1)}
                x return on intervention
              </span>
            </div>
          </div>

          {/* Formula & Governance Note */}
          <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800/80 text-xs text-slate-400 space-y-1 font-sans">
            <div className="text-[11px] font-mono text-slate-300">
              <strong className="text-sky-400">Transparent Formula:</strong> Net
              Value = (Outreach Patients × Model Precision × Assumed Reduction
              Rate × Cost per Readmission) - (Outreach Patients × Cost per
              Intervention)
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              *Disclaimer: This scenario calculator is an illustrative what-if
              estimation designed for health-system program budgeting. Actual
              clinical outcomes vary by health literacy, patient engagement, and
              intervention adherence.
            </p>
          </div>
        </div>
      )}

      {/* Flagged Patients Table */}
      <div className="rounded-2xl bg-slate-900/70 border border-slate-800 overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex justify-between items-center">
          <div>
            <h3 className="text-base font-semibold text-white">
              Highest-Risk Flagged Patients (Top 50 Displayed)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Ranked in real-time by calibrated 30-day readmission risk
              probability
            </p>
          </div>
          {loading && (
            <div className="flex items-center gap-2 text-xs text-sky-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Recalculating cohort...
            </div>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase font-mono tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Rank</th>
                <th className="py-3 px-4">Encounter ID</th>
                <th className="py-3 px-4">Calibrated Risk</th>
                <th className="py-3 px-4">Risk Tier</th>
                <th className="py-3 px-4">Demographics</th>
                <th className="py-3 px-4">Stay & Prior</th>
                <th className="py-3 px-4">Primary Dx</th>
                <th className="py-3 px-4">Observed Outcome</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {cohortData?.top_patients.map((p, idx) => {
                const isReadmit = p.readmitted_30d === 1;
                return (
                  <tr
                    key={p.encounter_id}
                    className="hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="py-3 px-4 font-mono text-slate-400">
                      #{idx + 1}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {p.encounter_id}
                    </td>
                    <td className="py-3 px-4 font-mono font-bold text-white text-sm">
                      {(p.risk_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                          p.risk_tier === "Critical"
                            ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                            : p.risk_tier === "High"
                              ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                              : p.risk_tier === "Medium"
                                ? "bg-sky-500/20 text-sky-300 border border-sky-500/30"
                                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                        }`}
                      >
                        {p.risk_tier}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300">
                      {p.age}, {p.gender}
                    </td>
                    <td className="py-3 px-4 text-slate-300">
                      {p.time_in_hospital}d stay, {p.number_inpatient} prior
                      inpt
                    </td>
                    <td className="py-3 px-4 text-slate-300">
                      {p.diag_1_group}
                    </td>
                    <td className="py-3 px-4">
                      {isReadmit ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Readmitted &lt;30d
                        </span>
                      ) : (
                        <span className="text-slate-400">Not Readmitted</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onSelectPatient(mapRowToPatientInput(p))}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 text-xs font-medium border border-sky-500/20 transition-colors"
                      >
                        Inspect
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
