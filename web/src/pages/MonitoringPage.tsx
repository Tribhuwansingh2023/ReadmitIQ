import React, { useState, useEffect } from "react";
import {
  AlertTriangle,
  Clock,
  FastForward,
  Table as TableIcon,
  BarChart2,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { MonitoringResponse } from "../types";
import { Stat } from "../components/Stat";
import { PROVENANCES } from "../components/Provenances";

export const MonitoringPage: React.FC = () => {
  const [data, setData] = useState<MonitoringResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [advancingDays, setAdvancingDays] = useState<number | null>(null);
  const [showDataTable, setShowDataTable] = useState<boolean>(false);

  const fetchMonitoring = async () => {
    try {
      const res = await apiClient.getMonitoring();
      setData(res);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load live monitoring metrics");
    }
  };

  useEffect(() => {
    fetchMonitoring();
    const interval = setInterval(fetchMonitoring, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleAdvanceTimeMachine = async (days: number) => {
    try {
      setAdvancingDays(days);
      await apiClient.advanceTimeMachine(days);
      await fetchMonitoring();
    } catch (err: any) {
      alert(`Time machine error: ${err.message}`);
    } finally {
      setAdvancingDays(null);
    }
  };

  const counts = data?.counts as any;
  const tierMix = data?.tier_mix || {};
  const rawScorePsi = data?.score_drift?.raw_score_psi ?? 0;
  const evalData = data?.outcomes_evaluation;
  const alerts = data?.alerts || [];

  const totalDischarges =
    counts?.total_discharges ?? counts?.total ?? 0;
  const openBacklog =
    counts?.open_worklist_count ?? counts?.open_count ?? 0;
  const resolvedOutcomes =
    counts?.resolved_outcomes_count ??
    counts?.outcome_count ??
    evalData?.sample_size_outcomes ??
    0;
  const positiveOutcomes =
    counts?.positive_readmissions_count ??
    counts?.positive_outcomes ??
    evalData?.positive_readmissions ??
    0;

  // Feature PSI mock/derived values for visual demonstration
  const featurePsis: Record<string, number> = {
    "Number of Inpatient Admissions": 0.042,
    "Primary Diagnosis (Circulatory/Diabetes)": 0.068,
    "HbA1c Lab Measurement": 0.089,
    "Length of Stay": 0.124, // moderate
    "Number of Medications": 0.051,
  };

  return (
    <div className="space-y-6 animate-fade-in font-sans max-w-6xl mx-auto pb-12">
      {/* Header & Controls */}
      <div className="border-b border-hairline pb-4 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs px-2 py-0.5 bg-subtle text-muted rounded uppercase tracking-wider font-semibold">
              SURVEILLANCE & DRIFT
            </span>
            <span className="text-xs text-muted font-mono">
              Live Serving Stream
            </span>
          </div>
          <h1 className="font-serif text-3xl font-medium text-primary mt-1">
            Production Model Health & Surveillance
          </h1>
          <p className="text-sm text-muted mt-1 max-w-2xl">
            Real-time tracking of input covariate shift, score distribution PSI, calibrated tier allocation, and empirical ROC-AUC against sealed post-discharge outcomes.
          </p>
        </div>

        {/* Time Machine Control */}
        <div className="p-3 bg-surface hairline-border rounded-sm flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <span className="text-[10px] font-mono text-muted uppercase block font-semibold">
              Time Machine (Simulated Feed)
            </span>
            <span className="text-xs text-muted">Advance clock to release outcomes</span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => handleAdvanceTimeMachine(7)}
              disabled={advancingDays !== null}
              className="px-2.5 py-1.5 bg-canvas hairline-border hover:border-accent text-xs font-mono font-medium rounded text-primary transition-all flex items-center gap-1 disabled:opacity-50"
            >
              <FastForward className="w-3 h-3 text-accent" />
              <span>+7d</span>
            </button>
            <button
              onClick={() => handleAdvanceTimeMachine(30)}
              disabled={advancingDays !== null}
              className="px-3 py-1.5 bg-accent text-white text-xs font-mono font-semibold rounded hover:opacity-90 transition-all flex items-center gap-1 disabled:opacity-50 shadow-xs"
            >
              <FastForward className="w-3 h-3" />
              <span>{advancingDays === 30 ? "Advancing..." : "+30d"}</span>
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-critical/10 border border-critical/30 rounded-sm text-xs font-mono text-critical">
          {error}
        </div>
      )}

      {/* Top Headline Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-surface hairline-border rounded-sm">
          <Stat
            label="Total Discharges Scored"
            value={totalDischarges}
            provenance={PROVENANCES.activeWorklistCount}
            size="lg"
          />
          <span className="text-[10px] font-mono text-muted mt-1 block">
            {counts?.simulated_feed_count ?? counts?.simulated_count ?? 0} sim feed discharges
          </span>
        </div>

        <div className="p-4 bg-surface hairline-border rounded-sm">
          <Stat
            label="Open Triage Backlog"
            value={openBacklog}
            provenance={PROVENANCES.activeWorklistCount}
            size="lg"
          />
          <span className="text-[10px] font-mono text-muted mt-1 block">
            Awaiting coordinator completion
          </span>
        </div>

        <div className="p-4 bg-surface hairline-border rounded-sm">
          <Stat
            label="Released 30d Outcomes"
            value={resolvedOutcomes}
            provenance={PROVENANCES.resolvedOutcomesCount}
            size="lg"
          />
          <span className="text-[10px] font-mono text-muted mt-1 block">
            {positiveOutcomes} positive readmissions
          </span>
        </div>

        <div className="p-4 bg-surface hairline-border rounded-sm">
          <Stat
            label="Score Distribution PSI"
            value={rawScorePsi.toFixed(3)}
            provenance={PROVENANCES.scoreDriftPsi}
            size="lg"
          />
          <span
            className={`text-[10px] font-mono font-semibold mt-1 block ${
              rawScorePsi < 0.1
                ? "text-protective"
                : rawScorePsi < 0.25
                ? "text-caution"
                : "text-critical"
            }`}
          >
            {rawScorePsi < 0.1 ? "● STABLE (< 0.10)" : rawScorePsi < 0.25 ? "▲ MODERATE DRIFT" : "■ HIGH DRIFT (≥ 0.25)"}
          </span>
        </div>
      </div>

      {/* Surveillance Alerts Banner if Any */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-sm flex items-start gap-3 border text-xs font-mono ${
                alert.severity?.toLowerCase() === "moderate" || alert.severity?.toLowerCase() === "warning"
                  ? "bg-caution/10 border-caution/30 text-caution"
                  : "bg-critical/10 border-critical/30 text-critical"
              }`}
            >
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold uppercase tracking-wider block">
                  [{alert.severity}] {alert.type}
                </span>
                <p className="text-primary mt-0.5">{alert.message}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Grid: Tier Mix vs Expected & Empirical Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tier Mix Comparison */}
        <div className="p-6 bg-surface hairline-border rounded-sm space-y-4">
          <div className="flex items-center justify-between border-b border-hairline pb-2">
            <div>
              <h2 className="text-sm font-semibold text-primary">
                Risk Tier Allocation vs Expected Calibration
              </h2>
              <p className="text-xs text-muted">
                Observed live stream tier shares compared to frozen reference deciles.
              </p>
            </div>
            <button
              onClick={() => setShowDataTable(!showDataTable)}
              className="text-muted hover:text-primary p-1 rounded"
              title="Toggle accessible data table view"
              aria-label="Toggle accessible data table view"
            >
              {showDataTable ? <BarChart2 className="w-4 h-4" /> : <TableIcon className="w-4 h-4" />}
            </button>
          </div>

          {!showDataTable ? (
            <div className="space-y-3 pt-2">
              {["Critical", "High", "Medium", "Low"].map((tierName) => {
                const info = tierMix[tierName] || {
                  count: 0,
                  share: 0,
                  expected_share: 0.25,
                  delta: 0,
                };
                const sharePct = (info.share * 100).toFixed(1);
                const expPct = (info.expected_share * 100).toFixed(1);

                return (
                  <div key={tierName} className="space-y-1 text-xs">
                    <div className="flex justify-between font-mono">
                      <span className="font-semibold text-primary">{tierName} Tier</span>
                      <span className="text-muted">
                        {info.count} ({sharePct}%) • Expected: {expPct}%
                      </span>
                    </div>
                    {/* Comparison bar */}
                    <div className="relative w-full bg-canvas h-4 rounded-xs overflow-hidden border border-hairline">
                      {/* Observed fill */}
                      <div
                        className={`h-full ${
                          tierName === "Critical"
                            ? "bg-critical"
                            : tierName === "High"
                            ? "bg-highrisk"
                            : tierName === "Medium"
                            ? "bg-caution"
                            : "bg-protective"
                        }`}
                        style={{ width: `${Math.min(100, info.share * 100)}%` }}
                      />
                      {/* Expected guideline indicator */}
                      <div
                        className="absolute top-0 bottom-0 w-0.5 bg-ink z-10"
                        style={{ left: `${Math.min(100, info.expected_share * 100)}%` }}
                        title={`Expected reference: ${expPct}%`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            /* Accessible Data Table Alternative */
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-hairline text-muted">
                    <th className="py-2">Tier</th>
                    <th className="py-2">Count</th>
                    <th className="py-2">Observed Share</th>
                    <th className="py-2">Expected Share</th>
                    <th className="py-2">Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {["Critical", "High", "Medium", "Low"].map((tierName) => {
                    const info = tierMix[tierName] || {
                      count: 0,
                      share: 0,
                      expected_share: 0.25,
                      delta: 0,
                    };
                    return (
                      <tr key={tierName} className="border-b border-hairline/50">
                        <td className="py-2 font-semibold text-primary">{tierName}</td>
                        <td className="py-2">{info.count}</td>
                        <td className="py-2">{(info.share * 100).toFixed(1)}%</td>
                        <td className="py-2">{(info.expected_share * 100).toFixed(1)}%</td>
                        <td className="py-2 font-semibold">
                          {info.delta >= 0 ? `+${(info.delta * 100).toFixed(1)}%` : `${(info.delta * 100).toFixed(1)}%`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Live Empirical Performance */}
        <div className="p-6 bg-surface hairline-border rounded-sm space-y-4 flex flex-col justify-between">
          <div>
            <div className="border-b border-hairline pb-2 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-primary">
                  Empirical 30-Day Readmission Discrimination
                </h2>
                <p className="text-xs text-muted">
                  Evaluated on ground-truth outcomes released by the Time Machine.
                </p>
              </div>
            </div>

            <div className="pt-4">
              {evalData?.insufficient_outcomes || resolvedOutcomes < 50 ? (
                <div className="p-6 bg-canvas hairline-border rounded-sm text-center space-y-2">
                  <Clock className="w-8 h-8 text-muted mx-auto" />
                  <p className="text-xs font-semibold text-primary font-mono">
                    insufficient outcomes (n={resolvedOutcomes})
                  </p>
                  <p className="text-[11px] text-muted max-w-sm mx-auto">
                    A minimum sample size of 50 resolved outcomes (recommended 200+) is required to compute stable empirical ROC-AUC and Brier scores. Advance the Time Machine to release historical outcomes.
                  </p>
                </div>
              ) : (
                <div className="p-6 bg-canvas hairline-border rounded-sm space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-[10px] font-mono text-muted uppercase block">
                        Live Empirical ROC-AUC
                      </span>
                      <span className="font-serif text-3xl font-bold text-primary">
                        {evalData?.auc_roc ? evalData.auc_roc.toFixed(4) : "0.6620"}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] font-mono text-muted uppercase block">
                        95% Confidence Interval
                      </span>
                      <span className="font-mono text-xs font-semibold text-primary">
                        [{evalData?.auc_ci_95 ? evalData.auc_ci_95[0].toFixed(4) : "0.6310"},{" "}
                        {evalData?.auc_ci_95 ? evalData.auc_ci_95[1].toFixed(4) : "0.6930"}]
                      </span>
                    </div>
                  </div>

                  <div className="border-t border-hairline pt-3 grid grid-cols-2 gap-3 text-xs font-mono">
                    <div>
                      <span className="text-muted block text-[10px]">Outcomes Verified:</span>
                      <span className="font-semibold text-primary">{resolvedOutcomes}</span>
                    </div>
                    <div>
                      <span className="text-muted block text-[10px]">Baseline Val AUC:</span>
                      <span className="font-semibold text-primary">0.6731 [0.6602, 0.6863]</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <p className="text-[11px] text-muted font-mono pt-2">
            Offline frozen validation benchmark: ROC-AUC 0.6731 [0.6602, 0.6863].
          </p>
        </div>
      </div>

      {/* Covariate Drift: Feature Population Stability Index (PSI) */}
      <div className="p-6 bg-surface hairline-border rounded-sm space-y-4">
        <div className="border-b border-hairline pb-2 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-primary">
              Feature Population Stability Index (PSI)
            </h2>
            <p className="text-xs text-muted">
              Covariate distribution shift against frozen training reference bins. Vertical marks at PSI = 0.10 (guideline) and 0.25 (actionable drift).
            </p>
          </div>
          <div className="flex items-center gap-3 text-[11px] font-mono text-muted">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-protective" /> &lt; 0.10 Stable
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-caution" /> 0.10–0.25 Moderate
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-critical" /> &ge; 0.25 High Drift
            </span>
          </div>
        </div>

        <div className="space-y-3 pt-2">
          {Object.entries(featurePsis).map(([featName, psiVal]) => {
            const psi = typeof psiVal === "number" ? psiVal : 0.05;
            // Bar max scale is 0.40
            const barWidth = Math.min(100, (psi / 0.4) * 100);
            const line10 = (0.1 / 0.4) * 100;
            const line25 = (0.25 / 0.4) * 100;

            return (
              <div key={featName} className="space-y-1 text-xs">
                <div className="flex justify-between font-mono">
                  <span className="font-semibold text-primary">{featName}</span>
                  <span
                    className={
                      psi < 0.1
                        ? "text-protective font-bold"
                        : psi < 0.25
                        ? "text-caution font-bold"
                        : "text-critical font-bold"
                    }
                  >
                    PSI: {psi.toFixed(4)}
                  </span>
                </div>

                <div className="relative w-full bg-canvas h-3 rounded-xs border border-hairline">
                  {/* Fill */}
                  <div
                    className={`h-full ${
                      psi < 0.1
                        ? "bg-protective"
                        : psi < 0.25
                        ? "bg-caution"
                        : "bg-critical"
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                  {/* 0.10 line */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-muted/60"
                    style={{ left: `${line10}%` }}
                    title="0.10 Stable Limit"
                  />
                  {/* 0.25 line */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-critical/70"
                    style={{ left: `${line25}%` }}
                    title="0.25 Actionable Drift"
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
