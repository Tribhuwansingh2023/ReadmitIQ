import React, { useState, useEffect } from "react";
import {
  Search,
  Filter,
  Play,
  Square,
  Upload,
  UserPlus,
  Sliders,
  CheckCircle2,
  PhoneCall,
  ChevronRight,
  RefreshCw,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { WorklistItem, WorklistResponse } from "../types";
import { RiskGlyph } from "../components/RiskGlyph";
import { Stat } from "../components/Stat";
import { Odometer } from "../components/Odometer";
import { PROVENANCES } from "../components/Provenances";

interface DeskPageProps {
  onOpenBrief: (dischargeId: number) => void;
  onOpenIntake: (tab?: "manual" | "csv" | "feed" | "api") => void;
}

export const DeskPage: React.FC<DeskPageProps> = ({
  onOpenBrief,
  onOpenIntake,
}) => {
  const [worklist, setWorklist] = useState<WorklistResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & State
  const [search, setSearch] = useState("");
  const [selectedTier, setSelectedTier] = useState<string>("");
  const [selectedStatus, setSelectedStatus] = useState<string>("open");
  const [capacity, setCapacity] = useState<number | null>(null);
  const [capacityInput, setCapacityInput] = useState<string>("15");
  const [focusedIndex, setFocusedIndex] = useState<number>(0);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Feed status
  const [feedRunning, setFeedRunning] = useState(false);

  // Load worklist
  const fetchWorklist = async (showLoading = false) => {
    if (showLoading) setIsRefreshing(true);
    try {
      const data = await apiClient.getWorklist({
        status: selectedStatus || undefined,
        tier: selectedTier || undefined,
        search: search || undefined,
        capacity: capacity !== null ? capacity : undefined,
        pageSize: 100,
      });
      setWorklist(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to fetch worklist");
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  };

  // Poll every 3 seconds for live arrivals
  useEffect(() => {
    fetchWorklist(false);
    const interval = setInterval(() => {
      fetchWorklist(false);
    }, 3000);
    return () => clearInterval(interval);
  }, [selectedTier, selectedStatus, search, capacity]);

  // Check feed status
  useEffect(() => {
    apiClient
      .getFeedStatus()
      .then((s) => setFeedRunning(s.is_running))
      .catch(() => {});
  }, []);

  // Keyboard navigation (j/k, Enter)
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (!worklist || worklist.items.length === 0) return;

      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        setFocusedIndex((prev) =>
          Math.min(prev + 1, worklist.items.length - 1)
        );
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        setFocusedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (worklist.items[focusedIndex]) {
          onOpenBrief(worklist.items[focusedIndex].id);
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [worklist, focusedIndex, onOpenBrief]);

  // Quick status toggle
  const handleQuickStatus = async (
    item: WorklistItem,
    newStatus: "contacted" | "closed"
  ) => {
    try {
      await apiClient.updateDischarge(item.id, {
        status: newStatus,
        actor: "Care Coordinator",
      });
      await apiClient.addAction(item.id, {
        type: newStatus === "contacted" ? "phone_call" : "case_closed",
        note:
          newStatus === "contacted"
            ? "Quick desk contact recorded."
            : "Encounter closed from triage desk.",
        actor: "Care Coordinator",
      });
      fetchWorklist(false);
    } catch (err: any) {
      alert(`Status update failed: ${err.message}`);
    }
  };

  const handleToggleFeed = async () => {
    try {
      if (feedRunning) {
        await apiClient.stopFeed();
        setFeedRunning(false);
      } else {
        await apiClient.startFeed(60, 42);
        setFeedRunning(true);
      }
      fetchWorklist(true);
    } catch (err: any) {
      alert(`Feed toggle error: ${err.message}`);
    }
  };

  const items = worklist?.items || [];
  const tierCounts = worklist?.tier_counts || {
    Critical: 0,
    High: 0,
    Medium: 0,
    Low: 0,
  };
  const totalCount = worklist?.total_count || 0;

  // Calculate summary counts
  const criticalOpen = tierCounts.Critical || 0;
  const highOpen = tierCounts.High || 0;
  const hasSimulated = items.some((i) => i.source === "simulated_feed");

  return (
    <div className="space-y-6 animate-fade-in font-sans">
      {/* Permanent simulation disclosure banner */}
      {hasSimulated && (
        <div
          role="status"
          className="bg-subtle/80 hairline-border px-4 py-2.5 rounded-sm flex items-center justify-between text-xs text-primary"
        >
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs px-1.5 py-0.5 bg-accent/15 text-accent rounded uppercase tracking-wider font-semibold">
              SIMULATED FEED ACTIVE
            </span>
            <span>
              Live replaying held-out historical diabetic inpatient encounters. True 30-day outcomes are sealed in ground-truth storage until Time Machine release.
            </span>
          </div>
          <button
            onClick={() => onOpenIntake("feed")}
            className="text-accent hover:underline font-medium text-xs whitespace-nowrap ml-4"
          >
            Feed Controls →
          </button>
        </div>
      )}

      {/* Screen Header & Editorial Description */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-hairline pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-serif font-bold text-primary tracking-tight">
              Discharge Desk
            </h1>
            <span className="text-xs font-mono px-2 py-0.5 rounded hairline-border bg-surface text-muted">
              Live Worklist
            </span>
            {isRefreshing && (
              <RefreshCw className="w-3.5 h-3.5 text-accent animate-spin" />
            )}
          </div>
          <p className="text-sm text-muted mt-1 max-w-2xl">
            Live patient encounters scored through frozen LightGBM + Structured RAG. Ranked strictly by raw model score to eliminate isotonic plateau ties.
          </p>
        </div>

        {/* Global Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleToggleFeed}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-sm text-xs font-medium hairline-border transition-colors ${
              feedRunning
                ? "bg-caution/15 text-caution hover:bg-caution/25"
                : "bg-surface hover:bg-subtle text-primary"
            }`}
          >
            {feedRunning ? (
              <>
                <Square className="w-3.5 h-3.5" /> Stop Feed
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 text-protective" /> Start Feed
              </>
            )}
          </button>

          <button
            onClick={() => onOpenIntake("csv")}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-surface hover:bg-subtle text-primary rounded-sm text-xs font-medium hairline-border transition-colors"
          >
            <Upload className="w-3.5 h-3.5 text-accent" /> Upload CSV
          </button>

          <button
            onClick={() => onOpenIntake("manual")}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent text-white hover:opacity-90 rounded-sm text-xs font-medium shadow-sm transition-opacity"
          >
            <UserPlus className="w-3.5 h-3.5" /> Add Patient
          </button>
        </div>
      </div>

      {/* Headline Odometers & Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat
          label="Awaiting Triage"
          value={<Odometer value={totalCount} />}
          unit="patients"
          provenance={PROVENANCES.activeWorklistCount}
          highlight="none"
        />
        <Stat
          label="Critical Risk (Top 5%)"
          value={<Odometer value={criticalOpen} />}
          unit="encounters"
          provenance={PROVENANCES.criticalOpenCount}
          highlight={criticalOpen > 0 ? "critical" : "none"}
        />
        <Stat
          label="High Risk (5–10%)"
          value={<Odometer value={highOpen} />}
          unit="encounters"
          provenance={PROVENANCES.highThreshold}
          highlight={highOpen > 0 ? "high" : "none"}
        />
        <Stat
          label="Resolved Outcomes"
          value={
            <Odometer
              value={items.filter((i) => i.outcome_readmitted_30d !== null).length}
            />
          }
          unit="verified"
          provenance={PROVENANCES.resolvedOutcomesCount}
          highlight="protective"
        />
      </div>

      {/* Triage Controls: Filters, Capacity Mode, Search */}
      <div className="bg-surface hairline-border p-3.5 rounded-sm flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 text-xs">
        {/* Tier filter chips */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-muted font-medium mr-1 uppercase text-[10px] tracking-wider">
            Tier:
          </span>
          {[
            { id: "", label: "All" },
            { id: "Critical", label: `Critical (${tierCounts.Critical || 0})` },
            { id: "High", label: `High (${tierCounts.High || 0})` },
            { id: "Medium", label: `Medium (${tierCounts.Medium || 0})` },
            { id: "Low", label: `Low (${tierCounts.Low || 0})` },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setSelectedTier(t.id)}
              className={`px-2.5 py-1 rounded-sm border transition-colors ${
                selectedTier === t.id
                  ? "bg-subtle text-primary font-semibold border-primary/40"
                  : "bg-surface text-muted hover:text-primary border-hairline"
              }`}
            >
              {t.label}
            </button>
          ))}

          <span className="text-hairline mx-1">|</span>

          {/* Status filter */}
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            aria-label="Filter encounters by status"
            className="bg-surface hairline-border text-primary px-2 py-1 rounded-sm text-xs focus:outline-none"
          >
            <option value="open">Open (Active)</option>
            <option value="new">Status: New</option>
            <option value="assigned">Status: Assigned</option>
            <option value="contacted">Status: Contacted</option>
            <option value="closed">Status: Closed</option>
            <option value="">All Statuses</option>
          </select>
        </div>

        {/* Capacity control & search */}
        <div className="flex items-center gap-3">
          {/* Capacity Slider / Toggle */}
          <div className="flex items-center gap-2 bg-subtle/50 px-2.5 py-1 rounded-sm border border-hairline">
            <Sliders className="w-3.5 h-3.5 text-accent" />
            <span className="text-muted text-[11px] whitespace-nowrap">
              Follow-up Capacity:
            </span>
            <input
              type="number"
              min="1"
              max="100"
              value={capacityInput}
              onChange={(e) => setCapacityInput(e.target.value)}
              aria-label="Follow-up patient capacity limit"
              className="w-12 bg-surface hairline-border px-1.5 py-0.5 rounded text-center font-mono text-xs text-primary"
            />
            <button
              onClick={() => {
                const n = parseInt(capacityInput, 10);
                if (capacity !== null) {
                  setCapacity(null);
                } else if (!isNaN(n) && n > 0) {
                  setCapacity(n);
                }
              }}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                capacity !== null
                  ? "bg-accent text-white"
                  : "bg-surface hover:bg-subtle text-muted hover:text-primary hairline-border"
              }`}
            >
              {capacity !== null ? `Top ${capacity} Active` : "Limit"}
            </button>
          </div>

          {/* Search box */}
          <div className="relative min-w-[180px]">
            <Search className="w-3.5 h-3.5 text-muted absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search ref or diag..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search worklist by MRN or diagnosis"
              className="w-full bg-surface hairline-border rounded-sm pl-8 pr-2.5 py-1 text-xs text-primary placeholder:text-muted focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-3 bg-critical/10 border border-critical/30 rounded-sm text-xs text-critical">
          {error}
        </div>
      )}

      {/* Worklist Table or Empty State */}
      {loading && !worklist ? (
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-16 bg-surface hairline-border rounded-sm skeleton" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="bg-surface hairline-border rounded-sm p-12 text-center space-y-4">
          <div className="w-12 h-12 mx-auto rounded-full bg-subtle flex items-center justify-center text-muted">
            <Filter className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-serif font-semibold text-primary">
              No discharges in current worklist view
            </h3>
            <p className="text-xs text-muted max-w-md mx-auto mt-1">
              Start the live simulated feed, upload an EHR discharge CSV batch, or add a patient encounter manually to begin clinical triage.
            </p>
          </div>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              onClick={() => handleToggleFeed()}
              className="px-4 py-2 bg-accent text-white rounded-sm text-xs font-medium hover:opacity-90"
            >
              Start Simulated Feed
            </button>
            <button
              onClick={() => onOpenIntake("csv")}
              className="px-4 py-2 bg-surface hover:bg-subtle text-primary rounded-sm text-xs font-medium hairline-border"
            >
              Upload CSV Batch
            </button>
            <button
              onClick={() => onOpenIntake("manual")}
              className="px-4 py-2 bg-surface hover:bg-subtle text-primary rounded-sm text-xs font-medium hairline-border"
            >
              Add Discharge Encounter
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-surface hairline-border rounded-sm overflow-hidden shadow-card">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-subtle/80 border-b border-hairline text-muted uppercase text-[10px] tracking-wider font-mono">
                  <th className="py-2.5 px-3 font-semibold">Priority & Rank</th>
                  <th className="py-2.5 px-3 font-semibold">Risk Tier & Scores</th>
                  <th className="py-2.5 px-3 font-semibold">Patient Clinical Profile</th>
                  <th className="py-2.5 px-3 font-semibold">Status & Assignee</th>
                  <th className="py-2.5 px-3 font-semibold">Source / Outcome</th>
                  <th className="py-2.5 px-3 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                {items.map((item, idx) => {
                  const isFocused = idx === focusedIndex;
                  const isCapacityPrioritized =
                    capacity !== null && idx < capacity;

                  return (
                    <tr
                      key={item.id}
                      tabIndex={0}
                      data-testid="patient-row"
                      onClick={() => {
                        setFocusedIndex(idx);
                        onOpenBrief(item.id);
                      }}
                      className={`cursor-pointer transition-colors group ${
                        isFocused
                          ? "bg-accent/5 ring-1 ring-inset ring-accent"
                          : "hover:bg-subtle/40"
                      } ${
                        isCapacityPrioritized
                          ? "border-l-4 border-l-accent"
                          : "border-l-4 border-l-transparent"
                      }`}
                    >
                      {/* Rank & ID */}
                      <td className="py-3 px-3 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[11px] text-muted w-6 text-center">
                            #{idx + 1}
                          </span>
                          <div>
                            <span className="font-mono font-semibold text-primary block">
                              {item.external_ref || `ENC-${item.id}`}
                            </span>
                            <span className="text-[10px] text-muted font-mono block">
                              ID: {item.id}
                            </span>
                          </div>
                        </div>
                      </td>

                      {/* Tier & Model Scores */}
                      <td className="py-3 px-3 whitespace-nowrap">
                        <div className="space-y-1">
                          <RiskGlyph tier={item.tier} size="sm" />
                          <div className="flex items-baseline gap-2 font-mono text-[11px]">
                            <span
                              className="font-semibold text-primary"
                              title="Clinician-facing calibrated readmission probability (Isotonic)"
                            >
                              P(cal): {(item.calibrated_prob * 100).toFixed(1)}%
                            </span>
                            <span
                              className="text-muted text-[10px]"
                              title="Raw model score used for sorting & tier assignment"
                            >
                              raw: {item.raw_score.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      </td>

                      {/* Clinical Facts */}
                      <td className="py-3 px-3">
                        <div className="space-y-0.5 max-w-xs">
                          <div className="font-medium text-primary">
                            {item.patient_summary.age} • {item.patient_summary.gender} •{" "}
                            <span className="text-accent">
                              {item.patient_summary.primary_diagnosis}
                            </span>
                          </div>
                          <div className="text-[11px] text-muted">
                            {item.patient_summary.time_in_hospital}d stay •{" "}
                            {item.patient_summary.number_inpatient} prior admissions
                          </div>
                        </div>
                      </td>

                      {/* Status & Assignee */}
                      <td className="py-3 px-3 whitespace-nowrap">
                        <div className="space-y-1">
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider ${
                              item.status === "closed"
                                ? "bg-subtle text-muted"
                                : item.status === "contacted"
                                ? "bg-protective/15 text-protective"
                                : item.status === "followup_scheduled"
                                ? "bg-accent/15 text-accent"
                                : "bg-caution/15 text-caution"
                            }`}
                          >
                            {item.status.replace("_", " ")}
                          </span>
                          <span className="text-[11px] text-muted block truncate max-w-[120px]">
                            {item.assignee || "Unassigned"}
                          </span>
                        </div>
                      </td>

                      {/* Source / Verified Outcome */}
                      <td className="py-3 px-3 whitespace-nowrap">
                        <div>
                          <span className="font-mono text-[10px] text-muted block uppercase">
                            {item.source}
                          </span>
                          {item.outcome_readmitted_30d !== null &&
                          item.outcome_readmitted_30d !== undefined ? (
                            <span
                              className={`inline-flex items-center gap-1 text-[11px] font-medium ${
                                item.outcome_readmitted_30d === 1
                                  ? "text-critical"
                                  : "text-protective"
                              }`}
                            >
                              {item.outcome_readmitted_30d === 1
                                ? "● Readmitted (30d)"
                                : "○ No Readmit"}
                            </span>
                          ) : (
                            <span className="text-[10px] text-muted italic">
                              Outcome pending
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Quick Actions */}
                      <td className="py-3 px-3 whitespace-nowrap text-right">
                        <div
                          className="flex items-center justify-end gap-1.5"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            onClick={() => handleQuickStatus(item, "contacted")}
                            className="p-1 rounded text-muted hover:text-protective hover:bg-subtle transition-colors"
                            title="Mark Contacted"
                          >
                            <PhoneCall className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleQuickStatus(item, "closed")}
                            className="p-1 rounded text-muted hover:text-primary hover:bg-subtle transition-colors"
                            title="Close Case"
                          >
                            <CheckCircle2 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => onOpenBrief(item.id)}
                            className="px-2 py-1 bg-subtle hover:bg-surface text-primary rounded hairline-border text-[11px] font-medium flex items-center gap-1"
                          >
                            <span>Brief</span>
                            <ChevronRight className="w-3 h-3 text-muted" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Table Footer with Keyboard Hint */}
          <div className="bg-subtle/40 px-4 py-2 border-t border-hairline flex items-center justify-between text-[11px] text-muted">
            <span>
              Showing {items.length} of {totalCount} total discharges
            </span>
            <div className="hidden sm:flex items-center gap-3">
              <span>
                Use <kbd className="px-1 py-0.5 bg-surface rounded hairline-border font-mono text-[10px]">j</kbd> /{" "}
                <kbd className="px-1 py-0.5 bg-surface rounded hairline-border font-mono text-[10px]">k</kbd> to navigate
              </span>
              <span>
                <kbd className="px-1 py-0.5 bg-surface rounded hairline-border font-mono text-[10px]">Enter</kbd> to open brief
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
