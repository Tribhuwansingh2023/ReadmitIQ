import React, { useState, useRef, useEffect } from "react";
import { Info } from "lucide-react";

export interface StatProvenance {
  source: string;
  howComputed: string;
  ci?: string;
  notes?: string;
  verifiedArtifactPath?: string;
}

interface StatProps {
  label: string;
  value: React.ReactNode;
  unit?: string;
  secondary?: React.ReactNode;
  provenance: StatProvenance;
  trend?: {
    direction: "up" | "down" | "neutral";
    label: string;
  };
  highlight?: "critical" | "high" | "caution" | "protective" | "none";
  size?: "sm" | "md" | "lg";
  variant?: "sm" | "md" | "lg" | "large";
}

export const Stat: React.FC<StatProps> = ({
  label,
  value,
  unit,
  secondary,
  provenance,
  trend,
  highlight = "none",
  size = "md",
  variant,
}) => {
  const effectiveSize = variant === "large" ? "lg" : variant || size;
  const [open, setOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  const highlightClass =
    highlight === "critical"
      ? "text-critical"
      : highlight === "high"
      ? "text-highrisk"
      : highlight === "caution"
      ? "text-caution"
      : highlight === "protective"
      ? "text-protective"
      : "text-primary";

  const sizeClass =
    effectiveSize === "lg"
      ? "text-4xl sm:text-5xl"
      : effectiveSize === "sm"
      ? "text-xl sm:text-2xl"
      : "text-2xl sm:text-3xl";

  return (
    <div className="relative group bg-surface hairline-border p-4 rounded-sm flex flex-col justify-between">
      <div className="flex items-center justify-between gap-1 mb-2">
        <span className="text-xs font-medium uppercase tracking-wider text-muted font-sans truncate">
          {label}
        </span>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="text-muted hover:text-primary transition-colors p-0.5 rounded focus-visible:ring-1 focus-visible:ring-protective"
          aria-label={`Inspect source provenance for ${label}`}
          title="Inspect data provenance"
        >
          <Info className="w-3.5 h-3.5" aria-hidden="true" />
        </button>
      </div>

      <div className="flex items-baseline gap-1.5 my-0.5">
        <span
          className={`font-mono tabular-nums font-semibold tracking-tight ${sizeClass} ${highlightClass}`}
        >
          {value}
        </span>
        {unit && <span className="text-sm font-sans text-muted">{unit}</span>}
      </div>

      {secondary && (
        <div className="text-xs font-sans text-muted mt-0.5 leading-snug">
          {secondary}
        </div>
      )}

      {trend && (
        <div className="mt-2 text-xs font-sans text-muted flex items-center gap-1">
          <span className="tabular-nums font-mono">{trend.label}</span>
        </div>
      )}

      {/* Provenance Popover */}
      {open && (
        <div
          ref={popoverRef}
          role="dialog"
          aria-label={`Provenance for ${label}`}
          className="absolute z-50 left-2 right-2 top-full mt-2 p-3 bg-surface hairline-border shadow-modal rounded-sm text-xs text-primary space-y-1.5 animate-fade-in font-sans"
        >
          <div className="flex items-center justify-between border-b border-hairline pb-1.5 font-semibold text-primary">
            <span>Data Provenance & Audit Trail</span>
            <button
              onClick={() => setOpen(false)}
              className="text-muted hover:text-primary text-xs"
            >
              ✕
            </button>
          </div>
          <div>
            <span className="text-muted font-mono block text-[10px] uppercase">Source:</span>
            <span className="font-mono text-[11px] text-accent break-all">
              {provenance.source}
            </span>
          </div>
          <div>
            <span className="text-muted font-mono block text-[10px] uppercase">Computation:</span>
            <p className="text-[11px] leading-relaxed">{provenance.howComputed}</p>
          </div>
          {provenance.ci && (
            <div>
              <span className="text-muted font-mono block text-[10px] uppercase">Confidence Interval:</span>
              <span className="font-mono text-[11px]">{provenance.ci}</span>
            </div>
          )}
          {provenance.verifiedArtifactPath && (
            <div className="pt-1 border-t border-hairline text-[10px] text-muted flex items-center justify-between font-mono">
              <span>Artifact verification:</span>
              <span className="text-protective">● Verified local file</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
