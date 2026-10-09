import React, { useEffect, useState } from 'react';
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
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { CohortPatient, CohortResponse, PatientFormInput } from '../types';

interface CohortPageProps {
  onSelectPatient: (patient: Partial<PatientFormInput>) => void;
}

export const CohortPage: React.FC<CohortPageProps> = ({ onSelectPatient }) => {
  const [capacity, setCapacity] = useState<number>(0.10);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [cohortData, setCohortData] = useState<CohortResponse | null>(null);

  const fetchCohort = async (cap: number) => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getCohort(cap);
      setCohortData(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load cohort replay');
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

  const mapRowToPatientInput = (p: CohortPatient): Partial<PatientFormInput> => {
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
            Simulate proactive care-team capacity allocation and measure true 30-day readmissions captured.
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
              of {cohortData.total_test_encounters.toLocaleString()} test encounters
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
              {(cohortData.recall * 100).toFixed(1)}% sensitivity ({cohortData.true_readmissions_caught} / {cohortData.total_true_readmissions_in_cohort})
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
              Intervention effort captures <strong className="text-white">{(cohortData.recall * 100).toFixed(1)}%</strong> of all readmissions at <strong className="text-white">{(capacity * 100).toFixed(0)}%</strong> workload
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
              style={{ width: `${100 - Math.min(100, cohortData.recall * 100)}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[11px] text-slate-500 mt-2">
            <span>0% caught</span>
            <span className="text-emerald-400 font-medium">Caught: {cohortData.true_readmissions_caught} patients</span>
            <span>100% ({cohortData.total_true_readmissions_in_cohort} total readmissions)</span>
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
              Ranked in real-time by calibrated 30-day readmission risk probability
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
                    <td className="py-3 px-4 font-mono text-slate-400">#{idx + 1}</td>
                    <td className="py-3 px-4 font-mono text-slate-300">{p.encounter_id}</td>
                    <td className="py-3 px-4 font-mono font-bold text-white text-sm">
                      {(p.risk_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                          p.risk_tier === 'Critical'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                            : p.risk_tier === 'High'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                            : p.risk_tier === 'Medium'
                            ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        }`}
                      >
                        {p.risk_tier}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300">
                      {p.age}, {p.gender}
                    </td>
                    <td className="py-3 px-4 text-slate-300">
                      {p.time_in_hospital}d stay, {p.number_inpatient} prior inpt
                    </td>
                    <td className="py-3 px-4 text-slate-300">{p.diag_1_group}</td>
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
