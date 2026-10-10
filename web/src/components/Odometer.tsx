import React, { useEffect, useState } from "react";

interface OdometerProps {
  value: number;
  duration?: number;
  className?: string;
}

export const Odometer: React.FC<OdometerProps> = ({
  value,
  duration = 400,
  className = "",
}) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    // Check reduced motion
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    if (prefersReducedMotion) {
      setDisplayValue(value);
      return;
    }

    const start = displayValue;
    const end = value;
    if (start === end) return;

    const startTime = performance.now();

    function update(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + (end - start) * eased);
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    const frameId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frameId);
  }, [value, duration]);

  return (
    <span className={`font-mono tabular-nums ${className}`}>
      {displayValue.toLocaleString()}
    </span>
  );
};
