import React, { useState, useEffect } from "react";
import {
  Search,
  LayoutDashboard,
  UserPlus,
  Play,
  Square,
  Activity,
  FileText,
  Network,
  Sun,
  Tv,
} from "lucide-react";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (screen: string) => void;
  onToggleTheme: () => void;
  onTogglePresentation: () => void;
  onStartFeed?: () => void;
  onStopFeed?: () => void;
  feedRunning?: boolean;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onNavigate,
  onToggleTheme,
  onTogglePresentation,
  onStartFeed,
  onStopFeed,
  feedRunning,
}) => {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

  const commands = [
    {
      id: "desk",
      title: "Go to Desk (Worklist)",
      icon: <LayoutDashboard className="w-4 h-4 text-accent" />,
      action: () => onNavigate("desk"),
    },
    {
      id: "intake-manual",
      title: "Intake: Add Discharge Encounter",
      icon: <UserPlus className="w-4 h-4 text-protective" />,
      action: () => onNavigate("intake"),
    },
    {
      id: "feed-toggle",
      title: feedRunning ? "Stop Simulated Feed" : "Start Simulated Feed",
      icon: feedRunning ? (
        <Square className="w-4 h-4 text-caution" />
      ) : (
        <Play className="w-4 h-4 text-protective" />
      ),
      action: () => {
        if (feedRunning) {
          onStopFeed?.();
        } else {
          onStartFeed?.();
        }
      },
    },
    {
      id: "monitoring",
      title: "Go to Live Monitoring Dashboard",
      icon: <Activity className="w-4 h-4 text-accent" />,
      action: () => onNavigate("monitoring"),
    },
    {
      id: "evidence",
      title: "Go to Scientific Evidence & Validation",
      icon: <FileText className="w-4 h-4 text-accent" />,
      action: () => onNavigate("evidence"),
    },
    {
      id: "integration",
      title: "Go to Enterprise Integration Architecture",
      icon: <Network className="w-4 h-4 text-accent" />,
      action: () => onNavigate("integration"),
    },
    {
      id: "theme",
      title: "Toggle Theme (Daylight / Night)",
      icon: <Sun className="w-4 h-4 text-caution" />,
      action: onToggleTheme,
    },
    {
      id: "presentation",
      title: "Toggle Presentation / Projector Mode (Key 'P')",
      icon: <Tv className="w-4 h-4 text-accent" />,
      action: onTogglePresentation,
    },
  ];

  const filtered = commands.filter((c) =>
    c.title.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (isOpen) onClose();
      }
      if (!isOpen) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % Math.max(1, filtered.length));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev === 0 ? filtered.length - 1 : prev - 1
        );
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (filtered[selectedIndex]) {
          filtered[selectedIndex].action();
          onClose();
        }
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filtered, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/40 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-surface hairline-border shadow-modal rounded-sm overflow-hidden animate-slide-down font-sans"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Command Palette"
      >
        <div className="flex items-center px-4 py-3 border-b border-hairline gap-3">
          <Search className="w-4 h-4 text-muted" aria-hidden="true" />
          <input
            type="text"
            className="w-full bg-transparent text-primary text-sm focus:outline-none placeholder:text-muted"
            placeholder="Type a command or jump to screen... (Esc to dismiss)"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            autoFocus
          />
          <kbd className="hidden sm:inline-block text-[10px] font-mono text-muted bg-subtle px-1.5 py-0.5 rounded hairline-border">
            ESC
          </kbd>
        </div>

        <div className="max-h-72 overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs text-muted">
              No matching commands.
            </div>
          ) : (
            filtered.map((cmd, idx) => (
              <button
                key={cmd.id}
                onClick={() => {
                  cmd.action();
                  onClose();
                }}
                onMouseEnter={() => setSelectedIndex(idx)}
                className={`w-full px-4 py-2.5 flex items-center justify-between text-xs text-left transition-colors ${
                  idx === selectedIndex
                    ? "bg-subtle text-primary font-medium"
                    : "text-muted hover:text-primary"
                }`}
              >
                <div className="flex items-center gap-3">
                  {cmd.icon}
                  <span>{cmd.title}</span>
                </div>
                {idx === selectedIndex && (
                  <kbd className="text-[10px] font-mono text-muted bg-surface px-1 py-0.5 rounded hairline-border">
                    ↵
                  </kbd>
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
