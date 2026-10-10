import React from "react";

interface EcgMonitorProps {
  isOnline: boolean;
  modelName?: string;
  version?: string;
}

export const EcgMonitor: React.FC<EcgMonitorProps> = ({
  isOnline,
  modelName = "LightGBM + Structured RAG",
  version = "1.0.0",
}) => {
  return (
    <div
      className="flex items-center gap-2.5 px-3 py-1.5 bg-surface hairline-border rounded-sm text-xs font-sans"
      role="status"
      aria-live="polite"
    >
      <div className="relative w-14 h-5 flex items-center justify-center overflow-hidden">
        {isOnline ? (
          <svg
            className="w-full h-full text-protective overflow-visible"
            viewBox="0 0 56 20"
            fill="none"
            aria-hidden="true"
          >
            {/* Cardiac rhythm path */}
            <path
              d="M0 10 h16 l3 -7 l4 15 l4 -11 l2 3 h27"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="animate-pulse-subtle"
            />
          </svg>
        ) : (
          <svg
            className="w-full h-full text-caution overflow-visible"
            viewBox="0 0 56 20"
            fill="none"
            aria-hidden="true"
          >
            {/* Isochronic flatline when disconnected */}
            <line
              x1="0"
              y1="10"
              x2="56"
              y2="10"
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray="2 2"
            />
          </svg>
        )}
      </div>

      <div className="hidden sm:flex flex-col leading-tight">
        <div className="flex items-center gap-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isOnline ? "bg-protective animate-ping" : "bg-caution"
            }`}
          />
          <span className="font-medium text-primary">
            {isOnline ? "Live Scoring Pipeline" : "API Offline"}
          </span>
        </div>
        <span className="text-[10px] font-mono text-muted truncate max-w-[160px]">
          {isOnline ? `${modelName} (v${version})` : "Retrying connection..."}
        </span>
      </div>
    </div>
  );
};
