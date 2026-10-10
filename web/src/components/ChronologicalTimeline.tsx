import React from "react";
import {
  Clock,
  UserCheck,
  CheckCircle2,
  AlertTriangle,
  FileText,
  PhoneCall,
  Home,
  Pill,
  Car,
  Activity,
  Award,
} from "lucide-react";

interface TimelineEventItem {
  id?: string | number;
  ts: string;
  type: string;
  title?: string;
  description?: string;
  actor?: string;
  note?: string;
  metadata?: Record<string, any>;
}

interface ChronologicalTimelineProps {
  events: TimelineEventItem[];
  dischargeId: number;
  outcomeReadmitted?: number | null;
}

export const ChronologicalTimeline: React.FC<ChronologicalTimelineProps> = ({
  events,
  dischargeId,
  outcomeReadmitted,
}) => {
  if (!events || events.length === 0) {
    return (
      <div className="p-6 bg-canvas hairline-border rounded-sm text-center text-xs text-muted font-mono">
        No chronological audit events recorded yet for encounter #{dischargeId}.
      </div>
    );
  }

  const getIconForType = (type: string) => {
    switch (type.toLowerCase()) {
      case "intake":
        return <Activity className="w-3.5 h-3.5 text-accent" />;
      case "status_change":
      case "status":
        return <UserCheck className="w-3.5 h-3.5 text-primary" />;
      case "task_completed":
      case "task_updated":
      case "task_created":
        return <CheckCircle2 className="w-3.5 h-3.5 text-protective" />;
      case "override":
        return <AlertTriangle className="w-3.5 h-3.5 text-critical" />;
      case "phone_call":
        return <PhoneCall className="w-3.5 h-3.5 text-accent" />;
      case "home_visit":
        return <Home className="w-3.5 h-3.5 text-caution" />;
      case "med_reconciliation":
      case "med_rec":
        return <Pill className="w-3.5 h-3.5 text-protective" />;
      case "transportation":
        return <Car className="w-3.5 h-3.5 text-accent" />;
      case "outcome":
        return <Award className="w-3.5 h-3.5 text-critical" />;
      case "outcome_pending":
        return <Clock className="w-3.5 h-3.5 text-muted" />;
      default:
        return <FileText className="w-3.5 h-3.5 text-muted" />;
    }
  };

  const getBadgeStyle = (type: string) => {
    switch (type.toLowerCase()) {
      case "outcome":
        return "bg-critical/10 text-critical border-critical/30";
      case "outcome_pending":
        return "bg-subtle text-muted border-hairline";
      case "task_completed":
        return "bg-protective/10 text-protective border-protective/30";
      case "override":
        return "bg-caution/10 text-caution border-caution/30";
      default:
        return "bg-canvas text-primary border-hairline";
    }
  };

  return (
    <div className="space-y-3 font-sans">
      <div className="flex items-center justify-between border-b border-hairline pb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] uppercase px-1.5 py-0.5 bg-accent/10 text-accent font-bold rounded">
            ActionLoop Stage 5
          </span>
          <h3 className="text-sm font-semibold text-primary">
            Closed-Loop Follow-up & Audit Trail
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {outcomeReadmitted !== undefined && outcomeReadmitted !== null && (
            <span
              className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                outcomeReadmitted === 1
                  ? "bg-critical/10 text-critical"
                  : "bg-protective/10 text-protective"
              }`}
            >
              {outcomeReadmitted === 1
                ? "30d Readmission Confirmed"
                : "Discharged Safely (No Readmit)"}
            </span>
          )}
          <span className="text-[11px] font-mono text-muted">
            {events.length} chronological events
          </span>
        </div>
      </div>

      <div className="relative pl-6 space-y-4 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-hairline">
        {events.map((evt, idx) => {
          const dateStr = evt.ts
            ? new Date(evt.ts).toLocaleString([], {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })
            : "Timestamp logged";

          const title = evt.title || evt.type.replace("_", " ").toUpperCase();
          const description = evt.description || evt.note || "";
          const actor = evt.actor || "System";

          return (
            <div key={evt.id || idx} className="relative group text-xs">
              {/* Node indicator */}
              <div className="absolute -left-6 top-1 w-5 h-5 rounded-full bg-surface hairline-border flex items-center justify-center shadow-2xs group-hover:border-accent">
                {getIconForType(evt.type)}
              </div>

              {/* Event card */}
              <div
                className={`p-3 rounded-sm border transition-colors ${getBadgeStyle(
                  evt.type
                )}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-primary">{title}</span>
                  <span className="font-mono text-[10px] text-muted">
                    {dateStr}
                  </span>
                </div>

                {description && (
                  <p className="text-[11px] text-muted mt-1 leading-relaxed">
                    {description}
                  </p>
                )}

                <div className="flex items-center justify-between gap-2 mt-2 pt-1.5 border-t border-hairline/50 text-[10px] font-mono text-muted">
                  <span>Actor: {actor}</span>
                  <span className="uppercase text-[9px] px-1 py-0.2 bg-subtle/80 rounded">
                    {evt.type}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
