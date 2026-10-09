import React, { useEffect, useState } from 'react';
import {
  Activity,
  BarChart2,
  LayoutDashboard,
  Scale,
  UserCheck,
} from 'lucide-react';
import { apiClient } from './api/client';
import { CohortPage } from './pages/CohortPage';
import { FairnessPage } from './pages/FairnessPage';
import { ModelLabPage } from './pages/ModelLabPage';
import { PatientPage } from './pages/PatientPage';
import type { HealthResponse, PatientFormInput } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'cohort' | 'patient' | 'lab' | 'fairness'>('cohort');
  const [selectedPatient, setSelectedPatient] = useState<Partial<PatientFormInput> | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const h = await apiClient.getHealth();
        setHealth(h);
        setApiConnected(true);
      } catch {
        setApiConnected(false);
      }
    };
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectCohortPatient = (patient: Partial<PatientFormInput>) => {
    setSelectedPatient(patient);
    setActiveTab('patient');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500 selection:text-slate-950">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 border border-sky-400/30">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-black tracking-tight text-white">ReadmitIQ</span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 font-mono">
                  v1.0 FROZEN
                </span>
              </div>
              <span className="text-[11px] text-slate-400 block -mt-0.5">
                Care-Team Hospital Readmission Command Center
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => setActiveTab('cohort')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'cohort'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <LayoutDashboard className="w-3.5 h-3.5" />
              Cohort Triage
            </button>

            <button
              onClick={() => setActiveTab('patient')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'patient'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <UserCheck className="w-3.5 h-3.5" />
              Patient Risk & RAG
            </button>

            <button
              onClick={() => setActiveTab('lab')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'lab'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              Model Lab & Ablation
            </button>

            <button
              onClick={() => setActiveTab('fairness')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'fairness'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Scale className="w-3.5 h-3.5" />
              Fairness & Drift
            </button>
          </nav>

          {/* Backend Status Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiConnected === true
                    ? 'bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse'
                    : apiConnected === false
                    ? 'bg-rose-500'
                    : 'bg-amber-400'
                }`}
              />
              <span className="text-slate-300">
                {apiConnected === true
                  ? 'API Online'
                  : apiConnected === false
                  ? 'API Disconnected'
                  : 'Connecting...'}
              </span>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Bar */}
        <div className="md:hidden flex overflow-x-auto px-4 py-2 bg-slate-950 border-t border-slate-800 gap-1 text-xs">
          <button
            onClick={() => setActiveTab('cohort')}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === 'cohort' ? 'bg-sky-500/20 text-sky-300' : 'text-slate-400'
            }`}
          >
            Cohort Triage
          </button>
          <button
            onClick={() => setActiveTab('patient')}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === 'patient' ? 'bg-sky-500/20 text-sky-300' : 'text-slate-400'
            }`}
          >
            Patient & RAG
          </button>
          <button
            onClick={() => setActiveTab('lab')}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === 'lab' ? 'bg-sky-500/20 text-sky-300' : 'text-slate-400'
            }`}
          >
            Model Lab
          </button>
          <button
            onClick={() => setActiveTab('fairness')}
            className={`px-3 py-1.5 rounded-lg whitespace-nowrap ${
              activeTab === 'fairness' ? 'bg-sky-500/20 text-sky-300' : 'text-slate-400'
            }`}
          >
            Fairness & Drift
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'cohort' && (
          <CohortPage onSelectPatient={handleSelectCohortPatient} />
        )}
        {activeTab === 'patient' && (
          <PatientPage initialPatient={selectedPatient} />
        )}
        {activeTab === 'lab' && <ModelLabPage />}
        {activeTab === 'fairness' && <FairnessPage />}
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 bg-slate-950/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-slate-400">ReadmitIQ</span>
            <span>•</span>
            <span className="font-mono">
              Model: {health?.model_name || 'LightGBM + Structured RAG'}
            </span>
            <span>•</span>
            <span className="font-mono">
              Git: {health?.git_commit_hash.slice(0, 7) || 'frozen'}
            </span>
          </div>
          <div className="text-center sm:text-right text-[11px] text-slate-500">
            Clinical Decision Support System. Not intended as sole autonomous diagnostic device.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
