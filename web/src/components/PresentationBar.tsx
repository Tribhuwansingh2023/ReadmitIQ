import React, { useState, useEffect } from "react";
import { Play, Pause, RotateCcw, X, Tv } from "lucide-react";

interface PresentationBarProps {
  currentScreen: string;
  onExit: () => void;
  onSelectScreen: (screen: string) => void;
}

export const PresentationBar: React.FC<PresentationBarProps> = ({
  currentScreen,
  onExit,
  onSelectScreen,
}) => {
  const [secondsLeft, setSecondsLeft] = useState(300); // 5:00
  const [isRunning, setIsRunning] = useState(true);

  useEffect(() => {
    let interval: any = null;
    if (isRunning && secondsLeft > 0) {
      interval = setInterval(() => {
        setSecondsLeft((s) => Math.max(0, s - 1));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRunning, secondsLeft]);

  const minutes = Math.floor(secondsLeft / 60);
  const seconds = secondsLeft % 60;
  const timeFormatted = `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;

  // Matching presenter cue from demo script
  const screenCues: Record<string, string> = {
    desk: "Cue 1: 'Here is our live discharge desk. Notice incoming discharges scored in real-time, ranked by raw model score to eliminate isotonic plateau ties.'",
    intake: "Cue 2: 'Intake channels: Real-time REST push, batch CSV drops with row validation, and our live simulated test replay.'",
    brief: "Cue 3: 'Discharge Brief: Model probability, log-odds SHAP drivers, and the Precedent Orbit showing similar historical cases.'",
    monitoring: "Cue 4: 'Live Monitoring: Population Stability Index (PSI) drift bars, tier mix shifts, and small-n outcome safety guards.'",
    evidence: "Cue 5: 'Scientific Evidence: Honest validation forest plots, paired bootstrap CIs, and subgroup demographic parity.'",
    integration: "Cue 6: 'Enterprise Integration: REST push, CSV batch drop, and FHIR Encounter resource mapper specification.'",
  };

  const cue =
    screenCues[currentScreen] ||
    "Scripted Demo Path: Start Feed -> Worklist -> Patient Detail -> Action -> Time Machine -> Monitoring -> Evidence.";

  return (
    <aside
      aria-label="Presentation & Demo Control Bar"
      className="fixed bottom-0 left-0 right-0 z-40 bg-surface/95 backdrop-blur hairline-border border-b-0 shadow-modal px-4 py-2.5 font-sans"
    >
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
        {/* Left: Indicator & Timer */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-2 py-0.5 bg-accent/15 text-accent rounded text-xs font-semibold">
            <Tv className="w-3.5 h-3.5" aria-hidden="true" />
            <span>PROJECTOR MODE (1280x720)</span>
          </div>

          <div className="flex items-center gap-2 bg-subtle px-2.5 py-1 rounded hairline-border font-mono text-xs">
            <span
              className={`font-semibold ${
                secondsLeft < 60 ? "text-critical animate-pulse" : "text-primary"
              }`}
            >
              {timeFormatted}
            </span>
            <button
              onClick={() => setIsRunning(!isRunning)}
              className="text-muted hover:text-primary transition-colors"
              title={isRunning ? "Pause timer" : "Start timer"}
            >
              {isRunning ? (
                <Pause className="w-3 h-3" />
              ) : (
                <Play className="w-3 h-3 text-protective" />
              )}
            </button>
            <button
              onClick={() => setSecondsLeft(300)}
              className="text-muted hover:text-primary transition-colors"
              title="Reset 5:00 timer"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Center: Live Presenter Cue */}
        <div className="flex-1 min-w-[280px] max-w-2xl px-3 py-1 bg-subtle/80 rounded border border-hairline/60 text-xs italic text-primary truncate">
          <span className="font-semibold not-italic text-accent mr-1">Cue:</span>
          {cue}
        </div>

        {/* Right: Screen Hotkeys & Exit */}
        <div className="flex items-center gap-2">
          <div className="hidden lg:flex items-center gap-1 text-[11px] font-mono text-muted">
            <button
              onClick={() => onSelectScreen("desk")}
              className={`px-1.5 py-0.5 rounded ${
                currentScreen === "desk" ? "bg-accent text-white" : "hover:text-primary"
              }`}
            >
              [1] Desk
            </button>
            <button
              onClick={() => onSelectScreen("intake")}
              className={`px-1.5 py-0.5 rounded ${
                currentScreen === "intake" ? "bg-accent text-white" : "hover:text-primary"
              }`}
            >
              [2] Intake
            </button>
            <button
              onClick={() => onSelectScreen("monitoring")}
              className={`px-1.5 py-0.5 rounded ${
                currentScreen === "monitoring" ? "bg-accent text-white" : "hover:text-primary"
              }`}
            >
              [4] Monitor
            </button>
            <button
              onClick={() => onSelectScreen("evidence")}
              className={`px-1.5 py-0.5 rounded ${
                currentScreen === "evidence" ? "bg-accent text-white" : "hover:text-primary"
              }`}
            >
              [5] Evidence
            </button>
          </div>

          <button
            onClick={onExit}
            className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-muted hover:text-primary bg-subtle hover:bg-surface rounded hairline-border transition-colors"
            title="Exit Presentation Mode (Press 'P')"
          >
            <X className="w-3.5 h-3.5" />
            <span>Exit (P)</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
