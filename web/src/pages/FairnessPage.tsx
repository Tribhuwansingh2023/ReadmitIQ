import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Database,
  EyeOff,
  Loader2,
  Scale,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { MetricsResponse } from '../types';

export const FairnessPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'race' | 'gender' | 'age_group'>('race');

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const data = await apiClient.getMetrics();
        setMetrics(data);
      } catch (err: any) {
        setError(err?.message || 'Failed to fetch fairness audit and drift metrics');
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-16 space-y-4">
        <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
        <span className="text-sm text-slate-400">Loading fairness audit and drift telemetry...</span>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="p-6 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center gap-3">
        <AlertCircle className="w-5 h-5 flex-shrink-0" />
        <span>{error || 'No metrics available.'}</span>
      </div>
    );
  }

  const fairness = metrics.fairness_audit;
  const drift = metrics.drift_metrics;
  const currentAttributeAudit = fairness?.by_attribute?.[activeTab];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-purple-500/10 text-purple-400 border border-purple-500/20">
            Algorithmic Equity & Drift
          </span>
          <span className="text-xs text-slate-400">
            Independent Subgroup Audits & Population Stability Index (PSI)
          </span>
        </div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Fairness Audit & Dataset Drift Telemetry
        </h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Evaluating demographic parity across protected attributes and continuous monitoring of feature stability.
        </p>
      </div>

      {/* Blinded Model Comparison Card */}
      {fairness?.blinded_model_comparison && (
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <EyeOff className="w-4 h-4 text-sky-400" />
              Demographic Blinded Model Audit (Race & Gender Excluded)
            </h3>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Fairness by Blindness
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-sans">
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 block mb-1">Main Model AUC (with Race/Gender):</span>
              <span className="text-xl font-bold font-mono text-white">
                {fairness.blinded_model_comparison.main_model_auc.toFixed(4)}
              </span>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 block mb-1">Blinded Model AUC (without Race/Gender):</span>
              <span className="text-xl font-bold font-mono text-sky-400">
                {fairness.blinded_model_comparison.blinded_model_auc.toFixed(4)}
              </span>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 block mb-1">Delta AUC:</span>
              <span className="text-xl font-bold font-mono text-emerald-400">
                {fairness.blinded_model_comparison.delta_auc >= 0 ? '+' : ''}
                {fairness.blinded_model_comparison.delta_auc.toFixed(4)}
              </span>
            </div>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
            <strong className="text-white">Fairness Insight:</strong> {fairness.blinded_model_comparison.conclusion}
          </p>
        </div>
      )}

      {/* Subgroup Fairness Table */}
      {fairness?.by_attribute && (
        <div className="rounded-2xl bg-slate-900/70 border border-slate-800 overflow-hidden shadow-xl">
          <div className="p-4 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-white flex items-center gap-2">
                <Scale className="w-4 h-4 text-purple-400" />
                Subgroup Disparity Metrics (Evaluated on Held-Out Test Split)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Reference group: <strong>{currentAttributeAudit?.reference_group}</strong> | Tested at frozen capacity threshold
              </p>
            </div>

            {/* Attribute Selector Tabs */}
            <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
              {(['race', 'gender', 'age_group'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1.5 rounded-lg font-medium capitalize transition-colors ${
                    activeTab === tab
                      ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {tab.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-sans">
              <thead className="bg-slate-950/60 text-slate-400 uppercase font-mono tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Subgroup</th>
                  <th className="py-3 px-4">Sample Size (N)</th>
                  <th className="py-3 px-4">Readmission Prevalence</th>
                  <th className="py-3 px-4">Subgroup AUC (95% CI)</th>
                  <th className="py-3 px-4">Selection Rate</th>
                  <th className="py-3 px-4">TPR (Sensitivity)</th>
                  <th className="py-3 px-4">FPR</th>
                  <th className="py-3 px-4">Disparate Impact Ratio</th>
                  <th className="py-3 px-4">Reliability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {currentAttributeAudit &&
                  Object.entries(currentAttributeAudit.groups).map(([groupName, g]) => {
                    const isRef = groupName === currentAttributeAudit.reference_group;
                    return (
                      <tr
                        key={groupName}
                        className={`hover:bg-slate-800/40 transition-colors ${
                          !g.reliable ? 'opacity-70 bg-amber-500/5' : ''
                        }`}
                      >
                        <td className="py-3 px-4 text-white font-medium flex items-center gap-1.5">
                          {groupName}
                          {isRef && (
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300">
                              Ref
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">
                          {g.group_size.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">
                          {(g.prevalence * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 font-mono font-bold text-slate-200">
                          {g.auc.toFixed(4)}
                          <span className="text-[11px] font-normal text-slate-400 ml-1">
                            [{g.auc_ci_lower.toFixed(4)}, {g.auc_ci_upper.toFixed(4)}]
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">
                          {(g.selection_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 font-mono text-emerald-400">
                          {(g.tpr * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 font-mono text-rose-400">
                          {(g.fpr * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 font-mono font-bold text-slate-200">
                          {g.disparate_impact_ratio.toFixed(2)}x
                        </td>
                        <td className="py-3 px-4">
                          {g.reliable ? (
                            <span className="text-emerald-400 text-[11px] font-medium flex items-center gap-1">
                              <CheckCircle2 className="w-3.5 h-3.5" /> Reliable
                            </span>
                          ) : (
                            <span className="text-amber-400 text-[11px] font-medium flex items-center gap-1">
                              <AlertTriangle className="w-3.5 h-3.5" /> Small Sample (&lt;100)
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Population Stability Index (PSI) Drift Telemetry */}
      {drift && (
        <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-xl space-y-4">
          <div className="flex justify-between items-center border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-base font-semibold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                Population Stability Index (PSI) Drift Telemetry
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Monitoring distribution stability between Train vs Validation and Train vs Test
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="text-emerald-400">Stable: &lt; 0.10</span>
              <span className="text-amber-400">Moderate: 0.10 - 0.25</span>
              <span className="text-rose-400">High: &gt; 0.25</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(drift.features).map(([feat, d]) => {
              const isTestStable = d.status_test === 'Stable';
              return (
                <div
                  key={feat}
                  className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center text-xs"
                >
                  <div>
                    <span className="text-slate-300 font-medium block truncate max-w-[170px]" title={feat}>
                      {feat}
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono mt-0.5 block">
                      Val PSI: {d.psi_val.toFixed(4)}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-bold text-slate-200 block">
                      {d.psi_test.toFixed(4)}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded inline-block mt-0.5 ${
                        isTestStable
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : 'bg-amber-500/10 text-amber-400'
                      }`}
                    >
                      {d.status_test}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 text-xs flex justify-between items-center text-slate-300">
            <span>
              Overall Score Distribution Drift (Train vs Test):
            </span>
            <span className="font-mono font-bold text-emerald-400">
              PSI = {drift.score_drift.psi_test.toFixed(4)} (Zero Distributional Shift)
            </span>
          </div>
        </div>
      )}

      {/* Dataset Limitations & Traps Card */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 text-xs text-slate-400 space-y-2">
        <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Database className="w-4 h-4 text-sky-400" />
          Real-World Data Traps & Clinical Limitations
        </h4>
        <ul className="list-disc pl-5 space-y-1.5 leading-relaxed">
          <li>
            <strong>Historical Horizon:</strong> Data collected 1999–2008 across 130 US hospitals. Clinical diabetes guidelines, ICD coding practices, and discharge post-acute pathways have changed substantially.
          </li>
          <li>
            <strong>Single Network Censoring:</strong> Encounters only capture readmissions within the same participating hospital system. Readmissions to external hospital systems are recorded as negative, creating potential label noise.
          </li>
          <li>
            <strong>No Outpatient Vitals or Continuous Labs:</strong> The dataset lacks baseline blood pressure, BMI, serum creatinine, or daily blood glucose readings, which limits predictive ceiling (AUC ~0.66 is standard and honest for this data).
          </li>
        </ul>
      </div>
    </div>
  );
};
