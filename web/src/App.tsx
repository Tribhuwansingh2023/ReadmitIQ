import React, { useEffect, useState } from "react";
import {
  Sun,
  Moon,
  Command,
  Monitor,
} from "lucide-react";
import { apiClient } from "./api/client";
import { DeskPage } from "./pages/DeskPage";
import { IntakePage } from "./pages/IntakePage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { EvidencePage } from "./pages/EvidencePage";
import { IntegrationPage } from "./pages/IntegrationPage";
import { DischargeBriefModal } from "./pages/DischargeBriefModal";
import { EcgMonitor } from "./components/EcgMonitor";
import { CommandPalette } from "./components/CommandPalette";
import { PresentationBar } from "./components/PresentationBar";

type ScreenTab = "desk" | "intake" | "monitoring" | "evidence" | "integration";

const parseTabFromHash = (): ScreenTab => {
  if (typeof window === "undefined") return "desk";
  const hash = window.location.hash.replace("#", "").toLowerCase();
  if (
    hash === "desk" ||
    hash === "intake" ||
    hash === "monitoring" ||
    hash === "evidence" ||
    hash === "integration"
  ) {
    return hash as ScreenTab;
  }
  return "desk";
};

export const App: React.FC = () => {
  const [activeTab, setActiveTabState] = useState<ScreenTab>(parseTabFromHash);
  const [theme, setTheme] = useState<"light" | "night">(() => {
    return (localStorage.getItem("readmitiq-theme") as "light" | "night") || "light";
  });
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [isPresentationMode, setIsPresentationMode] = useState<boolean>(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState<boolean>(false);
  const [selectedBriefId, setSelectedBriefId] = useState<number | null>(null);
  const [intakeInitialTab, setIntakeInitialTab] = useState<"manual" | "csv" | "feed" | "api">("manual");

  const setActiveTab = (tab: ScreenTab) => {
    setActiveTabState(tab);
    if (window.location.hash !== `#${tab}`) {
      window.location.hash = tab;
    }
  };

  useEffect(() => {
    const handleHash = () => {
      setActiveTabState(parseTabFromHash());
    };
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  // Sync theme
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "night") {
      root.classList.add("night");
      root.classList.remove("light");
    } else {
      root.classList.add("light");
      root.classList.remove("night");
    }
    localStorage.setItem("readmitiq-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "night" ? "light" : "night"));
  };

  // Heartbeat check for API
  const checkApiHealth = async () => {
    try {
      await apiClient.getHealth();
      setApiConnected(true);
    } catch {
      setApiConnected(false);
    }
  };

  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 4000);
    return () => clearInterval(interval);
  }, []);

  // Keyboard shortcut listener: P for presentation mode, 1-5 for screen switching
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (e.key === "p" || e.key === "P") {
        e.preventDefault();
        setIsPresentationMode((prev) => !prev);
      } else if (e.key === "1") {
        e.preventDefault();
        setActiveTab("desk");
      } else if (e.key === "2") {
        e.preventDefault();
        setActiveTab("intake");
      } else if (e.key === "3") {
        e.preventDefault();
        setActiveTab("monitoring");
      } else if (e.key === "4") {
        e.preventDefault();
        setActiveTab("evidence");
      } else if (e.key === "5") {
        e.preventDefault();
        setActiveTab("integration");
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleOpenIntakeWithTab = (tab?: "manual" | "csv" | "feed" | "api") => {
    if (tab) setIntakeInitialTab(tab);
    setActiveTab("intake");
  };

  return (
    <div className={`min-h-screen bg-canvas text-primary flex flex-col font-sans transition-colors duration-200 ${isPresentationMode ? "presentation-mode" : ""}`}>
      {/* Presentation Bar (Sticky at top when in presentation mode) */}
      {isPresentationMode && (
        <PresentationBar
          currentScreen={activeTab}
          onSelectScreen={(screen: string) => setActiveTab(screen as ScreenTab)}
          onExit={() => setIsPresentationMode(false)}
        />
      )}

      {/* Main Top Header */}
      <header className="border-b border-hairline bg-surface/90 backdrop-blur-xl sticky top-0 z-40 px-4 sm:px-6 py-3 transition-colors">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo & Clinical Brand */}
          <div className="flex items-center gap-3">
            {/* SVG Logo mark (ECG line forming IQ) */}
            <svg
              className="w-8 h-8 text-accent shrink-0"
              viewBox="0 0 100 100"
              fill="none"
              stroke="currentColor"
              strokeWidth="5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="50" cy="50" r="44" stroke="currentColor" strokeWidth="3" opacity="0.25" />
              <path d="M16 50 L32 50 L40 24 L48 76 L56 42 L64 54 L70 50 L84 50" />
            </svg>

            <div>
              <div className="flex items-center gap-2">
                <span className="font-serif text-xl font-bold tracking-tight text-primary">
                  Readmit<span className="text-accent italic font-normal">IQ</span>
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 bg-subtle text-muted rounded uppercase font-semibold">
                  Discharge Desk
                </span>
              </div>
              <p className="text-[11px] text-muted hidden sm:block">
                Clinical Care-Team Command Center • Frozen Serving Pipeline
              </p>
            </div>
          </div>

          {/* Nav Tabs for Desktop */}
          <nav aria-label="Main Navigation" className="hidden md:flex items-center gap-1 font-mono text-xs">
            {[
              { id: "desk", label: "Today's Discharges" },
              { id: "intake", label: "Intake Suite" },
              { id: "monitoring", label: "Live Monitoring" },
              { id: "evidence", label: "Evidence Dossier" },
              { id: "integration", label: "Integration" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as ScreenTab)}
                className={`px-3 py-1.5 rounded-sm transition-all ${
                  activeTab === tab.id
                    ? "bg-primary text-canvas font-bold shadow-xs"
                    : "text-muted hover:text-primary hover:bg-subtle"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Right Header Status & Tools */}
          <div className="flex items-center gap-3">
            {/* ECG status line */}
            <EcgMonitor isOnline={apiConnected === true} />

            {/* Command Palette Trigger */}
            <button
              onClick={() => setIsCommandPaletteOpen(true)}
              className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono text-muted hover:text-primary bg-canvas hairline-border rounded-sm hover:border-accent transition-all"
              title="Open Command Palette (Ctrl+K)"
            >
              <Command className="w-3.5 h-3.5" />
              <span>⌘K</span>
            </button>

            {/* Presentation Mode Toggle */}
            <button
              onClick={() => setIsPresentationMode(!isPresentationMode)}
              className={`p-1.5 rounded-sm border text-xs transition-colors hidden sm:flex items-center gap-1 font-mono ${
                isPresentationMode
                  ? "border-accent bg-accent/15 text-accent font-bold"
                  : "border-hairline hover:bg-subtle text-muted hover:text-primary"
              }`}
              title="Presentation Mode (P)"
            >
              <Monitor className="w-4 h-4" />
              <span className="text-[10px]">P</span>
            </button>

            {/* Daylight / Night Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-1.5 rounded-sm border border-hairline hover:bg-subtle text-muted hover:text-primary transition-colors"
              title={`Switch to ${theme === "light" ? "Night" : "Daylight"} theme`}
              aria-label={`Switch to ${theme === "light" ? "Night" : "Daylight"} theme`}
            >
              {theme === "night" ? (
                <Sun className="w-4 h-4 text-caution" />
              ) : (
                <Moon className="w-4 h-4 text-primary" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Screen Content View */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 pb-20 md:pb-8">
        {activeTab === "desk" && (
          <DeskPage
            onOpenBrief={(id) => setSelectedBriefId(id)}
            onOpenIntake={handleOpenIntakeWithTab}
          />
        )}

        {activeTab === "intake" && (
          <IntakePage
            initialTab={intakeInitialTab}
            onDischargeCreated={(newId) => {
              if (newId) {
                setSelectedBriefId(newId);
              }
              setActiveTab("desk");
            }}
          />
        )}

        {activeTab === "monitoring" && <MonitoringPage />}

        {activeTab === "evidence" && <EvidencePage />}

        {activeTab === "integration" && <IntegrationPage />}
      </main>

      {/* Patient Detail Modal (Brief) */}
      {selectedBriefId !== null && (
        <DischargeBriefModal
          dischargeId={selectedBriefId}
          onClose={() => setSelectedBriefId(null)}
          onStatusUpdated={() => {
            // refresh happens automatically via polling
          }}
        />
      )}

      {/* Command Palette Modal (Ctrl+K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onNavigate={(screen) => setActiveTab(screen as ScreenTab)}
        onToggleTheme={toggleTheme}
        onTogglePresentation={() => setIsPresentationMode((prev) => !prev)}
      />

      {/* Mobile Bottom Navigation Bar (for 390x844 and mobile viewports) */}
      <nav aria-label="Mobile Navigation" className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-surface border-t border-hairline px-2 py-1.5 flex justify-around font-mono text-[10px]">
        {[
          { id: "desk", label: "Desk" },
          { id: "intake", label: "Intake" },
          { id: "monitoring", label: "Monitor" },
          { id: "evidence", label: "Evidence" },
          { id: "integration", label: "Integrate" },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id as ScreenTab)}
            className={`py-1.5 px-2 rounded-xs flex flex-col items-center ${
              activeTab === item.id
                ? "text-accent font-bold bg-accent/10"
                : "text-muted hover:text-primary"
            }`}
          >
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
};

export default App;
