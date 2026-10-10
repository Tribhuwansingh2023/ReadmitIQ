import React, { useState, useEffect } from "react";
import {
  FileText,
  Upload,
  Terminal,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Download,
  Copy,
  Check,
  Play,
  Square,
  ArrowRight,
  Info,
} from "lucide-react";
import { apiClient, API_BASE_URL } from "../api/client";
import type { PatientFormInput, FeedStatusResponse } from "../types";

interface IntakePageProps {
  initialTab?: "manual" | "csv" | "feed" | "api";
  onDischargeCreated: (id?: number) => void;
}

const DEFAULT_PATIENT: PatientFormInput = {
  age: "[60-70)",
  gender: "Female",
  race: "Caucasian",
  time_in_hospital: 4,
  num_lab_procedures: 45,
  num_procedures: 1,
  num_medications: 15,
  number_outpatient: 0,
  number_emergency: 1,
  number_inpatient: 2,
  number_diagnoses: 8,
  admission_type_group: "Emergency",
  admission_source_group: "Emergency Room",
  discharge_disposition_group: "Home",
  medical_specialty: "InternalMedicine",
  payer_code: "Medicare",
  diag_1_group: "Circulatory",
  diag_2_group: "Diabetes",
  diag_3_group: "Other",
  max_glu_serum: "None",
  A1Cresult: ">8",
  diabetesMed: "Yes",
  change: "Ch",
  insulin: "Down",
  metformin: "Steady",
  glipizide: "No",
  glyburide: "No",
};

const PRESETS = [
  {
    name: "Critical Risk (Multi-Admit Elderly)",
    tag: "Critical",
    description: "75yo, 3 prior inpatient stays, circulatory dx, uncalibrated HbA1c >8, insulin titrated down",
    data: {
      ...DEFAULT_PATIENT,
      age: "[70-80)",
      time_in_hospital: 7,
      number_inpatient: 3,
      number_emergency: 2,
      num_medications: 24,
      diag_1_group: "Circulatory",
      A1Cresult: ">8",
      insulin: "Down",
      change: "Ch",
      discharge_disposition_group: "SNF",
    },
  },
  {
    name: "High Risk (Acute Respiratory & Polypharmacy)",
    tag: "High",
    description: "65yo, 5-day stay, 2 prior emergency visits, 22 medications, insulin change",
    data: {
      ...DEFAULT_PATIENT,
      age: "[60-70)",
      time_in_hospital: 5,
      number_inpatient: 1,
      number_emergency: 2,
      num_medications: 22,
      diag_1_group: "Respiratory",
      A1Cresult: "None",
      insulin: "Up",
      change: "Ch",
      discharge_disposition_group: "Home",
    },
  },
  {
    name: "Medium Risk (Emergency Diabetes)",
    tag: "Medium",
    description: "65yo, 1 emergency visit, acute diabetes, untested HbA1c, stable regimen",
    data: {
      ...DEFAULT_PATIENT,
      age: "[60-70)",
      time_in_hospital: 4,
      number_inpatient: 0,
      number_emergency: 1,
      num_medications: 14,
      diag_1_group: "Diabetes",
      A1Cresult: "None",
      insulin: "Steady",
      change: "No",
      discharge_disposition_group: "Home",
    },
  },
  {
    name: "Low Risk (Routine Stable)",
    tag: "Low",
    description: "45yo, 0 prior hospital visits, well-managed diabetes, routine discharge home",
    data: {
      ...DEFAULT_PATIENT,
      age: "[40-50)",
      time_in_hospital: 2,
      number_inpatient: 0,
      number_emergency: 0,
      number_outpatient: 0,
      num_medications: 7,
      diag_1_group: "Other",
      A1Cresult: "Norm",
      insulin: "No",
      metformin: "Steady",
      change: "No",
      discharge_disposition_group: "Home",
    },
  },
];

export const IntakePage: React.FC<IntakePageProps> = ({
  initialTab = "manual",
  onDischargeCreated,
}) => {
  const [activeTab, setActiveTab] = useState<"manual" | "csv" | "feed" | "api">(
    initialTab
  );

  // Manual Form State
  const [form, setForm] = useState<PatientFormInput>(DEFAULT_PATIENT);
  const [submitting, setSubmitting] = useState(false);
  const [manualSuccess, setManualSuccess] = useState<any | null>(null);
  const [manualError, setManualError] = useState<string | null>(null);

  // CSV Upload State
  const [dragOver, setDragOver] = useState(false);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvUploading, setCsvUploading] = useState(false);
  const [csvResult, setCsvResult] = useState<any | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvPreviewRows, setCsvPreviewRows] = useState<string[][]>([]);

  // Simulated Feed State
  const [feedStatus, setFeedStatus] = useState<FeedStatusResponse | null>(null);
  const [feedRate, setFeedRate] = useState<number>(60);
  const [feedSeed, setFeedSeed] = useState<number>(42);
  const [feedLoading, setFeedLoading] = useState(false);

  // Snippets Copied State
  const [copiedSnippet, setCopiedSnippet] = useState<string | null>(null);

  // Load feed status
  const refreshFeedStatus = async () => {
    try {
      const s = await apiClient.getFeedStatus();
      setFeedStatus(s);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    refreshFeedStatus();
    const interval = setInterval(refreshFeedStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleApplyPreset = (p: (typeof PRESETS)[0]) => {
    setForm(p.data);
    setManualSuccess(null);
    setManualError(null);
  };

  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setManualError(null);
    setManualSuccess(null);

    try {
      const res = await apiClient.createDischarge(form);
      setManualSuccess(res);
      setTimeout(() => {
        onDischargeCreated(res.discharge_id);
      }, 1500);
    } catch (err: any) {
      setManualError(err.message || "Failed to create encounter");
    } finally {
      setSubmitting(false);
    }
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectCsv(e.dataTransfer.files[0]);
    }
  };

  const handleSelectCsv = (file: File) => {
    setCsvFile(file);
    setCsvError(null);
    setCsvResult(null);

    // Read first few lines for preview
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = (event.target?.result as string) || "";
      const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
      const rows = lines.slice(0, 5).map((l) => l.split(","));
      setCsvPreviewRows(rows);
    };
    reader.readAsText(file.slice(0, 4096));
  };

  const handleCsvUpload = async () => {
    if (!csvFile) return;
    setCsvUploading(true);
    setCsvError(null);
    setCsvResult(null);

    try {
      const res = await apiClient.uploadCsv(csvFile);
      setCsvResult({
        imported_count: res.accepted_count ?? res.imported_count ?? 0,
        total_rows: res.total_rows_received ?? res.total_rows ?? 0,
        errors: (res.rejected_rows || res.errors || []).map((e: any) => ({
          row: e.row ?? e.index ?? 1,
          error: e.error || "Validation error",
        })),
      });
      setTimeout(() => {
        onDischargeCreated();
      }, 2000);
    } catch (err: any) {
      setCsvError(err.message || "CSV upload failed");
    } finally {
      setCsvUploading(false);
    }
  };

  const handleToggleFeed = async () => {
    setFeedLoading(true);
    try {
      if (feedStatus?.is_running) {
        const s = await apiClient.stopFeed();
        setFeedStatus(s);
      } else {
        const s = await apiClient.startFeed(feedRate, feedSeed);
        setFeedStatus(s);
      }
    } catch (err: any) {
      alert(`Feed error: ${err.message}`);
    } finally {
      setFeedLoading(false);
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(id);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  const curlSnippet = `curl -X POST "${API_BASE_URL}/v1/discharges" \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": "[70-80)",
    "gender": "Female",
    "race": "Caucasian",
    "time_in_hospital": 6,
    "num_lab_procedures": 48,
    "num_procedures": 1,
    "num_medications": 18,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 9,
    "admission_type_group": "Emergency",
    "admission_source_group": "Emergency Room",
    "discharge_disposition_group": "Home",
    "medical_specialty": "InternalMedicine",
    "payer_code": "Medicare",
    "diag_1_group": "Circulatory",
    "diag_2_group": "Diabetes",
    "diag_3_group": "Other",
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "diabetesMed": "Yes",
    "change": "Ch",
    "insulin": "Down",
    "metformin": "Steady"
  }'`;

  const pythonSnippet = `import requests

payload = {
    "age": "[70-80)",
    "gender": "Female",
    "race": "Caucasian",
    "time_in_hospital": 6,
    "num_lab_procedures": 48,
    "num_procedures": 1,
    "num_medications": 18,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 9,
    "admission_type_group": "Emergency",
    "admission_source_group": "Emergency Room",
    "discharge_disposition_group": "Home",
    "medical_specialty": "InternalMedicine",
    "payer_code": "Medicare",
    "diag_1_group": "Circulatory",
    "diag_2_group": "Diabetes",
    "diag_3_group": "Other",
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "diabetesMed": "Yes",
    "change": "Ch",
    "insulin": "Down",
    "metformin": "Steady",
}

response = requests.post("${API_BASE_URL}/v1/discharges", json=payload)
data = response.json()
print(f"Encounter ID: {data['discharge_id']}, Tier: {data['tier']}, Score: {data['raw_score']:.4f}")`;

  return (
    <div className="space-y-6 animate-fade-in font-sans max-w-6xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-hairline pb-4 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs px-2 py-0.5 bg-subtle text-muted rounded uppercase tracking-wider font-semibold">
              INTAKE SUITE
            </span>
          </div>
          <h1 className="font-serif text-3xl font-medium text-primary mt-1">
            Encounter Intake & Ingestion
          </h1>
          <p className="text-sm text-muted mt-1 max-w-2xl">
            Ingest inpatient discharges into the live triage desk via clinical manual entry, bulk CSV drops, automated EHR REST webhooks, or the controlled historical simulation feed.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-hairline gap-2 overflow-x-auto pb-px">
        <button
          onClick={() => setActiveTab("manual")}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all whitespace-nowrap ${
            activeTab === "manual"
              ? "border-accent text-accent font-semibold"
              : "border-transparent text-muted hover:text-primary"
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>Manual Entry & Presets</span>
        </button>

        <button
          onClick={() => setActiveTab("csv")}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all whitespace-nowrap ${
            activeTab === "csv"
              ? "border-accent text-accent font-semibold"
              : "border-transparent text-muted hover:text-primary"
          }`}
        >
          <Upload className="w-4 h-4" />
          <span>CSV Bulk Drop</span>
        </button>

        <button
          onClick={() => setActiveTab("api")}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all whitespace-nowrap ${
            activeTab === "api"
              ? "border-accent text-accent font-semibold"
              : "border-transparent text-muted hover:text-primary"
          }`}
        >
          <Terminal className="w-4 h-4" />
          <span>REST API / Webhook</span>
        </button>

        <button
          onClick={() => setActiveTab("feed")}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all whitespace-nowrap ${
            activeTab === "feed"
              ? "border-accent text-accent font-semibold"
              : "border-transparent text-muted hover:text-primary"
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Simulated Feed Panel</span>
          {feedStatus?.is_running && (
            <span className="w-2 h-2 rounded-full bg-protective animate-pulse" />
          )}
        </button>
      </div>

      {/* TAB 1: MANUAL FORM */}
      {activeTab === "manual" && (
        <div className="space-y-6">
          {/* Preset Buttons */}
          <div className="bg-surface p-4 rounded-sm hairline-border space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-muted">
                Clinical Benchmarking Presets
              </span>
              <span className="text-xs text-muted">Click to auto-populate the encounter</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  type="button"
                  onClick={() => handleApplyPreset(p)}
                  className="text-left p-3 rounded-sm border border-hairline hover:border-accent hover:bg-subtle transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="text-xs font-semibold text-primary">{p.name}</span>
                      <span
                        className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase font-bold ${
                          p.tag === "Critical"
                            ? "bg-critical/15 text-critical"
                            : p.tag === "High"
                            ? "bg-highrisk/15 text-highrisk"
                            : p.tag === "Medium"
                            ? "bg-caution/15 text-caution"
                            : "bg-protective/15 text-protective"
                        }`}
                      >
                        {p.tag}
                      </span>
                    </div>
                    <p className="text-[11px] text-muted line-clamp-2 leading-relaxed">
                      {p.description}
                    </p>
                  </div>
                  <span className="text-[10px] text-accent font-mono font-medium mt-2">
                    Apply preset →
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleManualSubmit} className="space-y-6 bg-surface p-6 rounded-sm hairline-border">
            {manualSuccess && (
              <div className="p-4 bg-protective/10 border border-protective/30 rounded-sm flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-protective" />
                  <div>
                    <p className="text-sm font-semibold text-primary">
                      Discharge Created Successfully!
                    </p>
                    <p className="text-xs text-muted font-mono">
                      Encounter #{manualSuccess.discharge_id} • Score: {manualSuccess.raw_score?.toFixed(4)} ({manualSuccess.tier} Tier)
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onDischargeCreated(manualSuccess.discharge_id)}
                  className="px-3 py-1.5 bg-protective text-white text-xs font-semibold rounded-sm hover:opacity-90 flex items-center gap-1"
                >
                  <span>Open Brief</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {manualError && (
              <div className="p-4 bg-critical/10 border border-critical/30 rounded-sm flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-critical" />
                <p className="text-xs text-critical font-mono">{manualError}</p>
              </div>
            )}

            {/* Field Groups */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Group 1: Demographics & Admission */}
              <div className="space-y-4">
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-muted border-b border-hairline pb-1">
                  1. Patient Demographics & Stay
                </h3>

                <div>
                  <label className="block text-xs font-medium text-primary mb-1">Age Bracket</label>
                  <select
                    value={form.age}
                    onChange={(e) => setForm({ ...form, age: e.target.value })}
                    className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                  >
                    {["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"].map(
                      (a) => (
                        <option key={a} value={a}>
                          {a}
                        </option>
                      )
                    )}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Gender</label>
                    <select
                      value={form.gender}
                      onChange={(e) => setForm({ ...form, gender: e.target.value })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                    >
                      <option value="Female">Female</option>
                      <option value="Male">Male</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Race</label>
                    <select
                      value={form.race}
                      onChange={(e) => setForm({ ...form, race: e.target.value })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                    >
                      <option value="Caucasian">Caucasian</option>
                      <option value="AfricanAmerican">AfricanAmerican</option>
                      <option value="Hispanic">Hispanic</option>
                      <option value="Asian">Asian</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-primary mb-1">
                    Length of Stay (Days: {form.time_in_hospital})
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="14"
                    value={form.time_in_hospital}
                    onChange={(e) => setForm({ ...form, time_in_hospital: parseInt(e.target.value) || 1 })}
                    className="w-full accent-accent"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-primary mb-1">Discharge Disposition</label>
                  <select
                    value={form.discharge_disposition_group}
                    onChange={(e) => setForm({ ...form, discharge_disposition_group: e.target.value })}
                    className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                  >
                    <option value="Home">Home (Routine)</option>
                    <option value="SNF">Skilled Nursing Facility (SNF)</option>
                    <option value="Other">Other Facility / Rehab</option>
                  </select>
                </div>
              </div>

              {/* Group 2: Prior Utilization & Diagnostic */}
              <div className="space-y-4">
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-muted border-b border-hairline pb-1">
                  2. Prior Encounters & Diagnostics
                </h3>

                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="block text-[11px] font-medium text-primary mb-1">Prior Inpat</label>
                    <input
                      type="number"
                      min="0"
                      max="15"
                      value={form.number_inpatient}
                      onChange={(e) => setForm({ ...form, number_inpatient: parseInt(e.target.value) || 0 })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2 py-1.5 font-mono text-primary"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-primary mb-1">Prior ER</label>
                    <input
                      type="number"
                      min="0"
                      max="15"
                      value={form.number_emergency}
                      onChange={(e) => setForm({ ...form, number_emergency: parseInt(e.target.value) || 0 })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2 py-1.5 font-mono text-primary"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-primary mb-1">Prior Outpat</label>
                    <input
                      type="number"
                      min="0"
                      max="15"
                      value={form.number_outpatient}
                      onChange={(e) => setForm({ ...form, number_outpatient: parseInt(e.target.value) || 0 })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2 py-1.5 font-mono text-primary"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-primary mb-1">Primary Diagnosis Category</label>
                  <select
                    value={form.diag_1_group}
                    onChange={(e) => setForm({ ...form, diag_1_group: e.target.value })}
                    className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                  >
                    <option value="Circulatory">Circulatory (Cardiac/Vascular)</option>
                    <option value="Respiratory">Respiratory (COPD/Pneumonia)</option>
                    <option value="Diabetes">Diabetes Primary</option>
                    <option value="Digestive">Digestive</option>
                    <option value="Injury">Injury</option>
                    <option value="Musculoskeletal">Musculoskeletal</option>
                    <option value="Genitourinary">Genitourinary</option>
                    <option value="Other">Other Medical</option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Lab Procedures</label>
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={form.num_lab_procedures}
                      onChange={(e) => setForm({ ...form, num_lab_procedures: parseInt(e.target.value) || 1 })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2 py-1.5 font-mono text-primary"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Diagnoses Count</label>
                    <input
                      type="number"
                      min="1"
                      max="16"
                      value={form.number_diagnoses}
                      onChange={(e) => setForm({ ...form, number_diagnoses: parseInt(e.target.value) || 1 })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2 py-1.5 font-mono text-primary"
                    />
                  </div>
                </div>
              </div>

              {/* Group 3: Medication & Glycemic */}
              <div className="space-y-4">
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-muted border-b border-hairline pb-1">
                  3. Glycemic Lab & Medications
                </h3>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">HbA1c Result</label>
                    <select
                      value={form.A1Cresult}
                      onChange={(e) => setForm({ ...form, A1Cresult: e.target.value })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                    >
                      <option value="None">None (Unmeasured)</option>
                      <option value="Norm">Norm (&lt; 7.0)</option>
                      <option value=">7">&gt; 7 (Elevated)</option>
                      <option value=">8">&gt; 8 (Poor Control)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Total Meds</label>
                    <input
                      type="number"
                      min="1"
                      max="60"
                      value={form.num_medications}
                      onChange={(e) => setForm({ ...form, num_medications: parseInt(e.target.value) || 1 })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2 py-1.5 font-mono text-primary"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Insulin Status</label>
                    <select
                      value={form.insulin}
                      onChange={(e) => setForm({ ...form, insulin: e.target.value })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                    >
                      <option value="No">No</option>
                      <option value="Steady">Steady</option>
                      <option value="Up">Up (Titrated)</option>
                      <option value="Down">Down (Titrated)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-primary mb-1">Regimen Change</label>
                    <select
                      value={form.change}
                      onChange={(e) => setForm({ ...form, change: e.target.value })}
                      className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                    >
                      <option value="Ch">Changed during stay</option>
                      <option value="No">No change</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-primary mb-1">Metformin Status</label>
                  <select
                    value={form.metformin}
                    onChange={(e) => setForm({ ...form, metformin: e.target.value })}
                    className="w-full text-xs bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
                  >
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="border-t border-hairline pt-4 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={() => setForm(DEFAULT_PATIENT)}
                className="px-4 py-2 text-xs font-medium text-muted hover:text-primary transition-colors"
              >
                Reset Fields
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="px-5 py-2.5 bg-accent text-white text-xs font-semibold rounded-sm hover:opacity-90 transition-opacity flex items-center gap-2 disabled:opacity-50"
              >
                {submitting ? "Scoring Encounter..." : "Score & Ingest Encounter"}
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>
      )}

      {/* TAB 2: CSV BULK DROP */}
      {activeTab === "csv" && (
        <div className="space-y-6">
          <div className="bg-surface p-6 rounded-sm hairline-border space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-hairline pb-3">
              <div>
                <h3 className="text-sm font-semibold text-primary">
                  Bulk Encounter Ingestion via CSV
                </h3>
                <p className="text-xs text-muted">
                  Upload daily discharge batch files with automated schema validation and immediate triage ranking.
                </p>
              </div>
              <a
                href={apiClient.getTemplateCsvUrl()}
                download="readmitiq_discharge_template.csv"
                className="inline-flex items-center gap-2 text-xs text-accent font-medium hover:underline py-1"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download CSV Template</span>
              </a>
            </div>

            {/* Drag and Drop Box */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleFileDrop}
              className={`border-2 border-dashed rounded-sm p-8 text-center transition-all ${
                dragOver
                  ? "border-accent bg-accent/5"
                  : "border-hairline hover:border-muted bg-canvas"
              }`}
            >
              <Upload className="w-8 h-8 text-muted mx-auto mb-2" />
              <p className="text-xs font-medium text-primary">
                Drag and drop your discharge CSV here, or{" "}
                <label className="text-accent hover:underline cursor-pointer">
                  browse files
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleSelectCsv(e.target.files[0]);
                    }}
                  />
                </label>
              </p>
              <p className="text-[11px] text-muted mt-1 font-mono">
                Supports up to 5,000 encounters per upload. UTF-8 encoded.
              </p>
            </div>

            {/* Selected File & Preview */}
            {csvFile && (
              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between p-3 bg-subtle rounded-sm text-xs">
                  <div className="flex items-center gap-2 font-mono">
                    <FileText className="w-4 h-4 text-accent" />
                    <span className="font-semibold text-primary">{csvFile.name}</span>
                    <span className="text-muted">({(csvFile.size / 1024).toFixed(1)} KB)</span>
                  </div>
                  <button
                    onClick={handleCsvUpload}
                    disabled={csvUploading}
                    className="px-4 py-1.5 bg-accent text-white text-xs font-semibold rounded-sm hover:opacity-90 disabled:opacity-50"
                  >
                    {csvUploading ? "Uploading & Scoring..." : "Validate & Import"}
                  </button>
                </div>

                {csvPreviewRows.length > 0 && (
                  <div className="border border-hairline rounded-sm overflow-hidden">
                    <div className="bg-subtle px-3 py-1.5 text-[11px] font-mono text-muted uppercase font-bold">
                      CSV File Preview (First {csvPreviewRows.length} Rows)
                    </div>
                    <div className="overflow-x-auto text-xs">
                      <table className="w-full text-left">
                        <tbody>
                          {csvPreviewRows.map((row, rIdx) => (
                            <tr
                              key={rIdx}
                              className={
                                rIdx === 0
                                  ? "bg-canvas border-b border-hairline font-semibold text-primary"
                                  : "border-b border-hairline/50 text-muted"
                              }
                            >
                              {row.slice(0, 8).map((cell, cIdx) => (
                                <td key={cIdx} className="p-2 font-mono text-[11px] whitespace-nowrap">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error Report */}
            {csvError && (
              <div className="p-4 bg-critical/10 border border-critical/30 rounded-sm space-y-1">
                <div className="flex items-center gap-2 text-critical font-semibold text-xs">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Validation Error</span>
                </div>
                <p className="text-xs font-mono text-critical">{csvError}</p>
              </div>
            )}

            {/* Result Report */}
            {csvResult && (
              <div className="p-4 bg-protective/10 border border-protective/30 rounded-sm space-y-2">
                <div className="flex items-center gap-2 text-protective font-semibold text-xs">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Batch Ingestion Completed</span>
                </div>
                <p className="text-xs text-primary font-mono">
                  Imported: {csvResult.imported_count || 0} discharges scored. (Total rows: {csvResult.total_rows || 0})
                </p>
                {csvResult.errors && csvResult.errors.length > 0 && (
                  <div className="mt-2 text-xs font-mono text-caution">
                    <p className="font-semibold">Row-level validation warnings:</p>
                    <ul className="list-disc list-inside mt-1">
                      {csvResult.errors.map((e: any, i: number) => (
                        <li key={i}>Row {e.row}: {e.error}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: REST API */}
      {activeTab === "api" && (
        <div className="space-y-6">
          <div className="bg-surface p-6 rounded-sm hairline-border space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-primary">
                Direct EHR REST Integration
              </h3>
              <p className="text-xs text-muted mt-1">
                Invoke the live scoring pipeline synchronously via POST <code className="font-mono text-accent">/v1/discharges</code>. Discharges are immediately calibrated, assigned risk tiers, and prioritized on the worklist.
              </p>
            </div>

            {/* Endpoint URL Box */}
            <div className="p-3 bg-canvas hairline-border rounded-sm flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="px-1.5 py-0.5 bg-accent/20 text-accent font-bold rounded uppercase">
                  POST
                </span>
                <span className="text-primary">{API_BASE_URL}/v1/discharges</span>
              </div>
              <span className="text-muted">Header: Content-Type: application/json</span>
            </div>

            {/* cURL Snippet */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold uppercase text-muted">
                  cURL Ingestion Snippet
                </span>
                <button
                  onClick={() => handleCopy(curlSnippet, "curl")}
                  className="flex items-center gap-1 text-xs text-accent hover:underline font-mono"
                >
                  {copiedSnippet === "curl" ? (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy cURL</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="p-3 bg-canvas text-primary hairline-border rounded-sm font-mono text-xs overflow-x-auto">
                {curlSnippet}
              </pre>
            </div>

            {/* Python Snippet */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold uppercase text-muted">
                  Python Requests Client Snippet
                </span>
                <button
                  onClick={() => handleCopy(pythonSnippet, "python")}
                  className="flex items-center gap-1 text-xs text-accent hover:underline font-mono"
                >
                  {copiedSnippet === "python" ? (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy Python</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="p-3 bg-canvas text-primary hairline-border rounded-sm font-mono text-xs overflow-x-auto">
                {pythonSnippet}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: SIMULATED FEED PANEL */}
      {activeTab === "feed" && (
        <div className="space-y-6">
          {/* Permanent Disclaimer Banner */}
          <div
            role="status"
            className="p-4 bg-caution/10 border border-caution/30 rounded-sm flex items-start gap-3"
          >
            <Info className="w-5 h-5 text-caution shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="text-xs font-semibold text-primary">
                Demonstration Disclosure Banner
              </p>
              <p className="text-xs text-primary leading-relaxed font-mono">
                simulated feed from held-out historical encounters, outcomes hidden
              </p>
              <p className="text-[11px] text-muted leading-relaxed">
                This feed streams real, de-identified diabetic encounters from the held-out split with timestamps shifted to present time. True 30-day readmission outcomes are stored locked and sealed until released via the Time Machine in the Live Monitoring dashboard.
              </p>
            </div>
          </div>

          <div className="bg-surface p-6 rounded-sm hairline-border space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-hairline pb-4">
              <div>
                <h3 className="text-sm font-semibold text-primary flex items-center gap-2">
                  <span>Simulated Feed Generator</span>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase ${
                      feedStatus?.is_running
                        ? "bg-protective/15 text-protective"
                        : "bg-subtle text-muted"
                    }`}
                  >
                    {feedStatus?.is_running ? "STREAM ACTIVE" : "STOPPED"}
                  </span>
                </h3>
                <p className="text-xs text-muted mt-1">
                  Replays historical inpatient encounters through the identical frozen model scoring code path.
                </p>
              </div>

              <button
                onClick={handleToggleFeed}
                disabled={feedLoading}
                className={`px-5 py-2.5 rounded-sm text-xs font-semibold flex items-center gap-2 transition-all ${
                  feedStatus?.is_running
                    ? "bg-critical text-white hover:opacity-90"
                    : "bg-accent text-white hover:opacity-90"
                } disabled:opacity-50`}
              >
                {feedStatus?.is_running ? (
                  <>
                    <Square className="w-4 h-4 fill-current" />
                    <span>Stop Simulation Stream</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Start Simulation Stream</span>
                  </>
                )}
              </button>
            </div>

            {/* Metrics & Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-4 bg-canvas hairline-border rounded-sm space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted">
                  Total Discharges Emitted
                </span>
                <p className="text-2xl font-mono font-bold text-primary">
                  {feedStatus?.total_emitted ?? 0}
                </p>
                <p className="text-[10px] text-muted font-mono">
                  Source: {feedStatus?.feed_source || "evaluation_stream"}
                </p>
              </div>

              <div className="p-4 bg-canvas hairline-border rounded-sm space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="font-medium text-primary">Emission Velocity</span>
                  <span className="font-mono text-accent font-bold">
                    {feedRate} / min
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="180"
                  step="10"
                  value={feedRate}
                  disabled={feedStatus?.is_running}
                  onChange={(e) => setFeedRate(parseInt(e.target.value) || 60)}
                  className="w-full accent-accent disabled:opacity-50"
                />
                <p className="text-[10px] text-muted">
                  Adjust pace of incoming discharge arrivals
                </p>
              </div>

              <div className="p-4 bg-canvas hairline-border rounded-sm space-y-2">
                <label className="block text-xs font-medium text-primary">
                  Randomization Seed
                </label>
                <input
                  type="number"
                  value={feedSeed}
                  disabled={feedStatus?.is_running}
                  onChange={(e) => setFeedSeed(parseInt(e.target.value) || 42)}
                  className="w-full text-xs bg-surface hairline-border rounded px-2.5 py-1.5 font-mono text-primary disabled:opacity-50"
                />
                <p className="text-[10px] text-muted">
                  Ensures deterministic reproduction across judge evaluations
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
