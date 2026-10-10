import React from "react";
import type { RiskTier } from "../types";

interface RiskGlyphProps {
  tier: RiskTier | string;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

export const RiskGlyph: React.FC<RiskGlyphProps> = ({
  tier,
  size = "md",
  showLabel = true,
}) => {
  const normTier = (tier || "Low").toLowerCase();

  let glyph = "○";
  let colorClass = "text-protective bg-protective/10 border-protective/30";
  let label = "Low (Bottom 80%)";

  if (normTier.includes("crit")) {
    glyph = "▲";
    colorClass = "text-critical bg-critical/10 border-critical/30";
    label = "Critical (Top 5%)";
  } else if (normTier.includes("high")) {
    glyph = "■";
    colorClass = "text-highrisk bg-highrisk/10 border-highrisk/30";
    label = "High (5–10%)";
  } else if (normTier.includes("med")) {
    glyph = "●";
    colorClass = "text-caution bg-caution/10 border-caution/30";
    label = "Medium (10–20%)";
  }

  const sizeClasses =
    size === "sm"
      ? "text-[11px] px-1.5 py-0.5 gap-1"
      : size === "lg"
      ? "text-sm px-3 py-1 gap-2"
      : "text-xs px-2 py-0.5 gap-1.5";

  return (
    <span
      className={`inline-flex items-center font-sans font-medium rounded-sm border ${colorClass} ${sizeClasses}`}
      title={label}
    >
      <span className="font-mono select-none" aria-hidden="true">
        {glyph}
      </span>
      {showLabel && <span>{tier}</span>}
      <span className="sr-only">Risk Tier: {tier}</span>
    </span>
  );
};
