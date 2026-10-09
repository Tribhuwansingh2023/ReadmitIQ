import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart2,
  HelpCircle,
  LayoutDashboard,
  Moon,
  RefreshCw,
  Scale,
  Sun,
  UserCheck,
  Zap,
} from "lucide-react";
import { apiClient } from "./api/client";
import { CohortPage } from "./pages/CohortPage";
import { FairnessPage } from "./pages/FairnessPage";
import { HowItWorksPage } from "./pages/HowItWorksPage";
import { ModelLabPage } from "./pages/ModelLabPage";
import { PatientPage } from "./pages/PatientPage";
import type { HealthResponse, PatientFormInput } from "./types";

const parseTabFromHash = ():
  | "cohort"
  | "patient"
  | "lab"
  | "fairness"
  | "how-it-works" => {
  if (typeof window === "undefined") return "cohort";
  const hash = window.location.hash.replace("#", "").toLowerCase();
  if (
    hash === "cohort" ||
    hash === "patient" ||
    hash === "lab" ||
    hash === "fairness" ||
    hash === "how-it-works"
  ) {
    return hash;
  }
  return "cohort";
};

export const App: React.FC = () => {
  const [activeTab, setActiveTabState] = useState<
    "cohort" | "patient" | "lab" | "fairness" | "how-it-works"
  >(parseTabFromHash);

  const setActiveTab = (
    tab: "cohort" | "patient" | "lab" | "fairness" | "how-it-works"
  ) => {
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

  const [selectedPatient, setSelectedPatient] =
    useState<Partial<PatientFormInput> | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (
      (localStorage.getItem("readmitiq-theme") as "dark" | "light") || "dark"
    );
  });
  const [demoMode, setDemoMode] = useState<boolean>(true);

  // Sync theme with document classList
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
      root.classList.remove("light");
    } else {
      root.classList.add("light");
      root.classList.remove("dark");
    }
    localStorage.setItem("readmitiq-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const checkApiHealth = async () => {
    try {
      const h = await apiClient.getHealth();
      setHealth(h);
      setApiConnected(true);
    } catch {
      setApiConnected(false);
    }
  };

  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectCohortPatient = (patient: Partial<PatientFormInput>) => {
    setSelectedPatient(patient);
    setActiveTab("patient");
  };

  return (
    <div className="min-h-screen flex flex-col font-sans transition-colors duration-200">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 bg-slate-950/85 dark:bg-slate-950/85 light:bg-white/90 backdrop-blur-xl border-b border-slate-800/80 dark:border-slate-800/80 light:border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 border border-sky-400/30 flex-shrink-0">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-black tracking-tight text-white dark:text-white light:text-slate-900">
                  ReadmitIQ
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 font-mono">
                  v1.0 FROZEN
                </span>
                <button
                  onClick={() => setDemoMode((prev) => !prev)}
                  title="Toggle Demo Mode"
                  className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border hidden sm:inline-flex items-center gap-1 transition-colors ${
                    demoMode
                      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                      : "bg-slate-800 text-slate-400 border-slate-700"
                  }`}
                >
                  <Zap className="w-2.5 h-2.5" />{" "}
                  {demoMode ? "Demo Mode" : "Standard Mode"}
                </button>
              </div>
              <span className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-500 block -mt-0.5">
                Clinical Care-Team Hospital Readmission Command Center
              </span>
            </div>
          </div>

          {/* Navigation Tabs (Desktop) */}
          <nav
            role="tablist"
            aria-label="Main Navigation"
            className="hidden lg:flex items-center gap-1 bg-slate-900/90 dark:bg-slate-900/90 light:bg-slate-100 p-1 rounded-xl border border-slate-800 dark:border-slate-800 light:border-slate-200 text-xs"
          >
            <button
              id="nav-tab-cohort"
              role="tab"
              aria-selected={activeTab === "cohort"}
              onClick={() => setActiveTab("cohort")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "cohort"
                  ? "bg-sky-500/20 text-sky-300 dark:text-sky-300 light:text-sky-600 border border-sky-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white dark:hover:text-white light:text-slate-600 light:hover:text-slate-900"
              }`}
            >
              <LayoutDashboard className="w-3.5 h-3.5" />
              Cohort Triage
            </button>

            <button
              id="nav-tab-patient"
              role="tab"
              aria-selected={activeTab === "patient"}
              onClick={() => setActiveTab("patient")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "patient"
                  ? "bg-sky-500/20 text-sky-300 dark:text-sky-300 light:text-sky-600 border border-sky-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white dark:hover:text-white light:text-slate-600 light:hover:text-slate-900"
              }`}
            >
              <UserCheck className="w-3.5 h-3.5" />
              Patient Risk & RAG
            </button>

            <button
              id="nav-tab-lab"
              role="tab"
              aria-selected={activeTab === "lab"}
              onClick={() => setActiveTab("lab")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "lab"
                  ? "bg-sky-500/20 text-sky-300 dark:text-sky-300 light:text-sky-600 border border-sky-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white dark:hover:text-white light:text-slate-600 light:hover:text-slate-900"
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              Model Lab & Ablation
            </button>

            <button
              id="nav-tab-fairness"
              role="tab"
              aria-selected={activeTab === "fairness"}
              onClick={() => setActiveTab("fairness")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "fairness"
                  ? "bg-sky-500/20 text-sky-300 dark:text-sky-300 light:text-sky-600 border border-sky-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white dark:hover:text-white light:text-slate-600 light:hover:text-slate-900"
              }`}
            >
              <Scale className="w-3.5 h-3.5" />
              Fairness & Drift
            </button>

            <button
              id="nav-tab-how-it-works"
              role="tab"
              aria-selected={activeTab === "how-it-works"}
              onClick={() => setActiveTab("how-it-works")}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "how-it-works"
                  ? "bg-sky-500/20 text-sky-300 dark:text-sky-300 light:text-sky-600 border border-sky-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white dark:hover:text-white light:text-slate-600 light:hover:text-slate-900"
              }`}
            >
              <HelpCircle className="w-3.5 h-3.5" />
              How It Works
            </button>
          </nav>

          {/* Right Controls: Theme Toggle & Backend Status */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Theme Toggle Button */}
            <button
              id="btn-theme-toggle"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              className="p-2 rounded-xl bg-slate-900 dark:bg-slate-900 light:bg-slate-100 border border-slate-800 dark:border-slate-800 light:border-slate-200 text-slate-300 dark:text-slate-300 light:text-slate-700 hover:text-white transition-colors"
            >
              {theme === "dark" ? (
                <Sun className="w-4 h-4 text-amber-400" />
              ) : (
                <Moon className="w-4 h-4 text-sky-600" />
              )}
            </button>

            {/* Backend Status Badge */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 dark:bg-slate-900 light:bg-slate-100 border border-slate-800 dark:border-slate-800 light:border-slate-200 text-xs font-mono">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiConnected === true
                    ? "bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse"
                    : apiConnected === false
                      ? "bg-rose-500"
                      : "bg-amber-400"
                }`}
              />
              <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">
                {apiConnected === true
                  ? "API Online"
                  : apiConnected === false
                    ? "API Offline"
                    : "Connecting..."}
              </span>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Bar */}
        <div className="lg:hidden flex overflow-x-auto px-4 py-2 bg-slate-950 dark:bg-slate-950 light:bg-slate-100 border-t border-slate-800 dark:border-slate-800 light:border-slate-200 gap-1 text-xs">
          <button
            id="mobile-tab-cohort"
            onClick={() => setActiveTab("cohort")}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === "cohort"
                ? "bg-sky-500/20 text-sky-300 font-semibold"
                : "text-slate-400"
            }`}
          >
            Cohort Triage
          </button>
          <button
            id="mobile-tab-patient"
            onClick={() => setActiveTab("patient")}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === "patient"
                ? "bg-sky-500/20 text-sky-300 font-semibold"
                : "text-slate-400"
            }`}
          >
            Patient & RAG
          </button>
          <button
            id="mobile-tab-lab"
            onClick={() => setActiveTab("lab")}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === "lab"
                ? "bg-sky-500/20 text-sky-300 font-semibold"
                : "text-slate-400"
            }`}
          >
            Model Lab
          </button>
          <button
            id="mobile-tab-fairness"
            onClick={() => setActiveTab("fairness")}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === "fairness"
                ? "bg-sky-500/20 text-sky-300 font-semibold"
                : "text-slate-400"
            }`}
          >
            Fairness & Drift
          </button>
          <button
            id="mobile-tab-how-it-works"
            onClick={() => setActiveTab("how-it-works")}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === "how-it-works"
                ? "bg-sky-500/20 text-sky-300 font-semibold"
                : "text-slate-400"
            }`}
          >
            How It Works
          </button>
        </div>

        {/* Unreachable API Warning Banner */}
        {apiConnected === false && (
          <div className="bg-rose-500/15 border-b border-rose-500/30 px-4 py-2 text-xs text-rose-300 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0" />
              <span>
                <strong>API Server Unreachable:</strong> Cannot connect to
                backend server at http://localhost:8000. Please start the
                backend (
                <code className="font-mono bg-rose-500/20 px-1 py-0.5 rounded">
                  uvicorn api.main:app
                </code>
                ) or check connection.
              </span>
            </div>
            <button
              onClick={checkApiHealth}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 font-medium text-xs border border-rose-500/30 transition-colors whitespace-nowrap"
            >
              <RefreshCw className="w-3 h-3" />
              Retry Connection
            </button>
          </div>
        )}
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === "cohort" && (
          <CohortPage onSelectPatient={handleSelectCohortPatient} />
        )}
        {activeTab === "patient" && (
          <PatientPage initialPatient={selectedPatient} />
        )}
        {activeTab === "lab" && <ModelLabPage />}
        {activeTab === "fairness" && <FairnessPage />}
        {activeTab === "how-it-works" && <HowItWorksPage />}
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 dark:border-slate-800/80 light:border-slate-200 bg-slate-950/60 dark:bg-slate-950/60 light:bg-slate-100 py-6 text-xs text-slate-500 dark:text-slate-500 light:text-slate-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-slate-400 dark:text-slate-400 light:text-slate-700">
              ReadmitIQ
            </span>
            <span>•</span>
            <span className="font-mono">
              Model: {health?.model_name || "LightGBM + Structured RAG"}
            </span>
            <span>•</span>
            <span className="font-mono">
              Git: {health?.git_commit_hash.slice(0, 7) || "frozen"}
            </span>
          </div>
          <div className="text-center sm:text-right text-[11px]">
            Clinical Decision Support System. Not intended as sole autonomous
            diagnostic device.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
