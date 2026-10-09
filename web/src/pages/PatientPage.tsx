import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  FileText,
  History,
  Loader2,
  ShieldAlert,
  Zap,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type {
  PatientFormInput,
  PredictResponse,
} from '../types';

interface PatientPageProps {
  initialPatient?: Partial<PatientFormInput> | null;
}

const DEFAULT_PATIENT: PatientFormInput = {
  age: '[60-70)',
  gender: 'Female',
  race: 'Caucasian',
  time_in_hospital: 4,
  num_lab_procedures: 45,
  num_procedures: 1,
  num_medications: 15,
  number_outpatient: 0,
  number_emergency: 1,
  number_inpatient: 2,
  number_diagnoses: 8,
  admission_type_group: 'Emergency',
  admission_source_group: 'Emergency Room',
  discharge_disposition_group: 'Home',
  medical_specialty: 'InternalMedicine',
  payer_code: 'Medicare',
  diag_1_group: 'Circulatory',
  diag_2_group: 'Diabetes',
  diag_3_group: 'Other',
  max_glu_serum: 'None',
  A1Cresult: '>8',
  diabetesMed: 'Yes',
  change: 'Ch',
  insulin: 'Down',
  metformin: 'Steady',
};

const PRESETS: { name: string; description: string; data: PatientFormInput }[] = [
  {
    name: 'High Risk (Multi-Admit Elderly)',
    description: '75yo, 3 prior inpatient stays, circulatory dx, uncalibrated HbA1c >8, insulin titrated',
    data: {
      ...DEFAULT_PATIENT,
      age: '[70-80)',
      time_in_hospital: 7,
      number_inpatient: 3,
      number_emergency: 2,
      num_medications: 24,
      diag_1_group: 'Circulatory',
      A1Cresult: '>8',
      insulin: 'Down',
      change: 'Ch',
      discharge_disposition_group: 'SNF',
    },
  },
  {
    name: 'Medium Risk (Emergency Diabetes)',
    description: '65yo, 1 prior emergency visit, newly diagnosed acute diabetes, untested HbA1c',
    data: {
      ...DEFAULT_PATIENT,
      age: '[60-70)',
      time_in_hospital: 4,
      number_inpatient: 0,
      number_emergency: 1,
      num_medications: 14,
      diag_1_group: 'Diabetes',
      A1Cresult: 'None',
      insulin: 'Steady',
      change: 'No',
      discharge_disposition_group: 'Home',
    },
  },
  {
    name: 'Low Risk (Routine Stable)',
    description: '45yo, 0 prior hospital visits, well-managed diabetes, routine discharge home',
    data: {
      ...DEFAULT_PATIENT,
      age: '[40-50)',
      time_in_hospital: 2,
      number_inpatient: 0,
      number_emergency: 0,
      number_outpatient: 0,
      num_medications: 7,
      diag_1_group: 'Other',
      A1Cresult: 'Norm',
      insulin: 'No',
      metformin: 'Steady',
      change: 'No',
      discharge_disposition_group: 'Home',
    },
  },
];

export const PatientPage: React.FC<PatientPageProps> = ({ initialPatient }) => {
  const [formData, setFormData] = useState<PatientFormInput>(DEFAULT_PATIENT);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialPatient) {
      setFormData((prev) => ({ ...prev, ...initialPatient }));
    }
  }, [initialPatient]);

  const handleInputChange = (field: keyof PatientFormInput, val: any) => {
    setFormData((prev) => ({ ...prev, [field]: val }));
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.predict(formData);
      setPredictResult(res);
    } catch (err: any) {
      setError(err?.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  // Run initial prediction on mount
  useEffect(() => {
    handlePredict();
  }, []);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-purple-500/10 text-purple-400 border border-purple-500/20">
              Real-Time Inference
            </span>
            <span className="text-xs text-slate-400">
              Calibrated LightGBM + FAISS Precedent Retrieval
            </span>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Patient Risk Profile & Clinical Precedent RAG
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Enter encounter attributes or load clinical archetypes to evaluate readmission risk and explainable precedents.
          </p>
        </div>

        {/* Presets */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-400 flex items-center gap-1 mr-1">
            <Zap className="w-3.5 h-3.5 text-amber-400" /> Presets:
          </span>
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => {
                setFormData(preset.data);
              }}
              className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 text-xs font-medium text-slate-200 border border-slate-700/60 transition-colors shadow-sm"
            >
              {preset.name.split(' ')[0]}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid: Form Left, Results Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Clinical Form */}
        <div className="lg:col-span-5 p-6 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-xl space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <FileText className="w-4 h-4 text-sky-400" />
              Encounter Attributes
            </h3>
            <span className="text-xs text-slate-500">UCI Standard Schema</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {/* Age */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Age Band</label>
              <select
                value={formData.age}
                onChange={(e) => handleInputChange('age', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              >
                {['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'].map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </div>

            {/* Gender */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Gender</label>
              <select
                value={formData.gender}
                onChange={(e) => handleInputChange('gender', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>

            {/* Time in hospital */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Length of Stay (Days)</label>
              <input
                type="number"
                min="1"
                max="14"
                value={formData.time_in_hospital}
                onChange={(e) => handleInputChange('time_in_hospital', parseInt(e.target.value) || 1)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>

            {/* Prior Inpatient */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Prior Inpatient Visits (1y)</label>
              <input
                type="number"
                min="0"
                max="20"
                value={formData.number_inpatient}
                onChange={(e) => handleInputChange('number_inpatient', parseInt(e.target.value) || 0)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500 font-semibold text-amber-300"
              />
            </div>

            {/* Prior Emergency */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Prior Emergency Visits (1y)</label>
              <input
                type="number"
                min="0"
                max="20"
                value={formData.number_emergency}
                onChange={(e) => handleInputChange('number_emergency', parseInt(e.target.value) || 0)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>

            {/* Num Medications */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Medications Count</label>
              <input
                type="number"
                min="1"
                max="100"
                value={formData.num_medications}
                onChange={(e) => handleInputChange('num_medications', parseInt(e.target.value) || 1)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              />
            </div>

            {/* Primary Diagnosis */}
            <div className="col-span-2">
              <label className="text-slate-400 font-medium block mb-1">Primary Diagnosis (ICD-9 Group)</label>
              <select
                value={formData.diag_1_group}
                onChange={(e) => handleInputChange('diag_1_group', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500 font-medium"
              >
                {['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'].map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>

            {/* Discharge Destination */}
            <div className="col-span-2">
              <label className="text-slate-400 font-medium block mb-1">Discharge Destination</label>
              <select
                value={formData.discharge_disposition_group}
                onChange={(e) => handleInputChange('discharge_disposition_group', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              >
                {['Home', 'Home Health', 'SNF', 'Other Inpatient', 'Left AMA', 'Other'].map((dest) => (
                  <option key={dest} value={dest}>{dest}</option>
                ))}
              </select>
            </div>

            {/* HbA1c Result */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">HbA1c Lab Test</label>
              <select
                value={formData.A1Cresult}
                onChange={(e) => handleInputChange('A1Cresult', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              >
                <option value="None">None (Untested)</option>
                <option value="Norm">Norm (&lt;7%)</option>
                <option value=">7">&gt;7%</option>
                <option value=">8">&gt;8% (Elevated)</option>
              </select>
            </div>

            {/* Insulin */}
            <div>
              <label className="text-slate-400 font-medium block mb-1">Insulin Regimen</label>
              <select
                value={formData.insulin}
                onChange={(e) => handleInputChange('insulin', e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-sky-500"
              >
                <option value="No">No</option>
                <option value="Steady">Steady</option>
                <option value="Up">Up (Titrated)</option>
                <option value="Down">Down (Titrated)</option>
              </select>
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-3 px-4 rounded-xl bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 font-bold text-sm flex items-center justify-center gap-2 transition-colors shadow-lg shadow-sky-500/20 cursor-pointer"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing Precedents & Calculating Risk...
              </>
            ) : (
              <>
                <Activity className="w-4 h-4" />
                Run Calibrated Risk Prediction
              </>
            )}
          </button>
        </div>

        {/* Right Column: Prediction Results & RAG Precedents */}
        <div className="lg:col-span-7 space-y-6">
          {predictResult && (
            <>
              {/* Risk Gauge & Tier Card */}
              <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl relative overflow-hidden">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
                  {/* Score Gauge */}
                  <div className="flex items-center gap-5">
                    <div className="relative w-28 h-28 flex items-center justify-center">
                      <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          className="stroke-slate-800 stroke-[10] fill-none"
                        />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          className={`stroke-[10] fill-none transition-all duration-700 ${
                            predictResult.risk_tier === 'Critical'
                              ? 'stroke-rose-500'
                              : predictResult.risk_tier === 'High'
                              ? 'stroke-amber-500'
                              : predictResult.risk_tier === 'Medium'
                              ? 'stroke-sky-500'
                              : 'stroke-emerald-500'
                          }`}
                          strokeDasharray={251.2}
                          strokeDashoffset={
                            251.2 - (251.2 * Math.min(1, predictResult.calibrated_risk_probability))
                          }
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute text-center">
                        <span className="text-2xl font-black font-mono text-white">
                          {(predictResult.calibrated_risk_probability * 100).toFixed(1)}%
                        </span>
                        <span className="block text-[10px] text-slate-400 font-medium">Risk Score</span>
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center gap-2">
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                            predictResult.risk_tier === 'Critical'
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                              : predictResult.risk_tier === 'High'
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                              : predictResult.risk_tier === 'Medium'
                              ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          }`}
                        >
                          {predictResult.risk_tier} Risk Tier
                        </span>
                        {predictResult.predicted_readmission && (
                          <span className="px-2 py-0.5 rounded text-[11px] bg-rose-500/10 text-rose-400 border border-rose-500/20 font-medium">
                            Flagged for Outreach
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 mt-2 max-w-sm">
                        Decision threshold: <strong>{(predictResult.decision_threshold * 100).toFixed(1)}%</strong>. 
                        Top-10% capacity threshold: <strong>{(predictResult.capacity_thresholds.top_10pct_high * 100).toFixed(1)}%</strong>.
                      </p>
                    </div>
                  </div>

                  {/* Precedent Neighborhood KPI */}
                  <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 text-center min-w-[170px]">
                    <div className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">
                      RAG Precedents
                    </div>
                    <div className="text-2xl font-mono font-bold text-sky-400 mt-1">
                      {(predictResult.similar_patients_summary.neighbor_readmission_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-[11px] text-slate-500 mt-1">
                      Readmission rate across 20 historical matches
                    </div>
                  </div>
                </div>
              </div>

              {/* Explainability: SHAP Clinical Factors */}
              <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h3 className="text-base font-semibold text-white flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4 text-amber-400" />
                    Plain-Language Contributing Factors (SHAP)
                  </h3>
                  <span className="text-xs text-slate-500 font-mono">TreeExplainer</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Risk Increasing */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-rose-400 flex items-center gap-1">
                      <ArrowUpRight className="w-3.5 h-3.5" /> Risk Increasing Drivers
                    </span>
                    {predictResult.top_risk_increasing_factors.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-rose-500/5 border border-rose-500/20 text-xs text-slate-200"
                      >
                        <div className="font-medium text-rose-300">{item.explanation}</div>
                        <div className="text-[11px] text-slate-400 mt-1 flex justify-between">
                          <span>SHAP +{item.shap_value.toFixed(3)}</span>
                          <span className="font-mono text-slate-500">{item.feature}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Risk Reducing */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1">
                      <ArrowDownRight className="w-3.5 h-3.5" /> Risk Reducing Drivers
                    </span>
                    {predictResult.top_risk_reducing_factors.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20 text-xs text-slate-200"
                      >
                        <div className="font-medium text-emerald-300">{item.explanation}</div>
                        <div className="text-[11px] text-slate-400 mt-1 flex justify-between">
                          <span>SHAP {item.shap_value.toFixed(3)}</span>
                          <span className="font-mono text-slate-500">{item.feature}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* RAG Precedents Table */}
              <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-base font-semibold text-white flex items-center gap-2">
                      <History className="w-4 h-4 text-sky-400" />
                      Top Similar Historical Precedents (TRAIN Corpus)
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Matched via 64-dimensional SVD cosine similarity in FAISS index (N = 69,535 historical cases)
                    </p>
                  </div>
                  <span className="text-xs font-mono text-sky-400">
                    k = 20 Neighborhood
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950/60 text-slate-400 uppercase font-mono tracking-wider border-b border-slate-800">
                      <tr>
                        <th className="py-2.5 px-3">Encounter ID</th>
                        <th className="py-2.5 px-3">Similarity</th>
                        <th className="py-2.5 px-3">Patient Profile</th>
                        <th className="py-2.5 px-3">Stay & Visits</th>
                        <th className="py-2.5 px-3">Primary Diagnosis</th>
                        <th className="py-2.5 px-3">Historical Outcome</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-sans">
                      {predictResult.similar_patients_summary.top_neighbors.map((sim) => {
                        const isReadmit = sim.readmitted_30d === 1;
                        return (
                          <tr key={sim.encounter_id} className="hover:bg-slate-800/30">
                            <td className="py-2.5 px-3 font-mono text-slate-400">
                              {sim.encounter_id}
                            </td>
                            <td className="py-2.5 px-3 font-mono font-semibold text-sky-400">
                              {(sim.similarity_score * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {sim.age}, {sim.gender}
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {sim.time_in_hospital}d stay, {sim.prior_inpatient} inpt
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {sim.primary_diagnosis}
                            </td>
                            <td className="py-2.5 px-3">
                              {isReadmit ? (
                                <span className="text-rose-400 font-semibold flex items-center gap-1">
                                  <AlertTriangle className="w-3 h-3" /> Readmitted
                                </span>
                              ) : (
                                <span className="text-slate-400 flex items-center gap-1">
                                  <CheckCircle2 className="w-3 h-3 text-slate-500" /> Not Readmitted
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
