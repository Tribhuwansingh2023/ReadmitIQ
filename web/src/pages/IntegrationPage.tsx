import React, { useState } from "react";
import {
  Server,
  Layers,
  Cpu,
  Lock,
  Copy,
  Check,
  FileCode,
} from "lucide-react";
import { API_BASE_URL } from "../api/client";

export const IntegrationPage: React.FC = () => {
  const [copiedSnippet, setCopiedSnippet] = useState<string | null>(null);

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(id);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  const curlExample = `curl -X POST "${API_BASE_URL}/v1/discharges" \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": "[60-70)",
    "gender": "Female",
    "race": "Caucasian",
    "time_in_hospital": 4,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 8,
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

  const tsExample = `import { apiClient } from "./api/client";

// Ingest a live encounter from clinical system
const response = await apiClient.createDischarge({
  age: "[70-80)",
  gender: "Male",
  race: "Caucasian",
  time_in_hospital: 6,
  num_medications: 22,
  number_inpatient: 2,
  diag_1_group: "Circulatory",
  A1Cresult: ">8",
  insulin: "Down",
  change: "Ch",
});

console.log(\`Scored encounter #\${response.discharge_id}: tier=\${response.tier}\`);`;

  return (
    <div className="space-y-6 animate-fade-in font-sans max-w-6xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-hairline pb-4 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs px-2 py-0.5 bg-subtle text-muted rounded uppercase tracking-wider font-semibold">
              ARCHITECTURE & INTEROPERABILITY
            </span>
            <span className="text-xs text-muted font-mono">Enterprise Readiness</span>
          </div>
          <h1 className="font-serif text-3xl font-medium text-primary mt-1">
            System Integration & Deployment Architecture
          </h1>
          <p className="text-sm text-muted mt-1 max-w-2xl">
            Technical blueprint for connecting ReadmitIQ into hospital health systems, electronic health record (EHR) platforms, and enterprise data warehouses.
          </p>
        </div>
      </div>

      {/* Integration Modes Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Architecture Mode 1 */}
        <div className="p-5 bg-surface hairline-border rounded-sm space-y-3 relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono px-2 py-0.5 bg-protective/15 text-protective rounded font-bold uppercase">
              IMPLEMENTED & LIVE
            </span>
            <Server className="w-5 h-5 text-accent" />
          </div>
          <h3 className="font-serif text-lg font-semibold text-primary">
            REST Webhook Push
          </h3>
          <p className="text-xs text-muted leading-relaxed">
            Direct synchronous JSON HTTP push triggered on patient discharge event. Evaluated and placed on live triage desk in &lt; 25 ms.
          </p>
          <div className="border-t border-hairline pt-2 text-[11px] font-mono text-muted space-y-1">
            <div>Endpoint: <code>POST /v1/discharges</code></div>
            <div>Latency: 1.89 ms per encounter</div>
            <div>Status: Production Ready</div>
          </div>
        </div>

        {/* Architecture Mode 2 */}
        <div className="p-5 bg-surface hairline-border rounded-sm space-y-3 relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono px-2 py-0.5 bg-protective/15 text-protective rounded font-bold uppercase">
              IMPLEMENTED & LIVE
            </span>
            <FileCode className="w-5 h-5 text-accent" />
          </div>
          <h3 className="font-serif text-lg font-semibold text-primary">
            Batch CSV Drop
          </h3>
          <p className="text-xs text-muted leading-relaxed">
            Automated SFTP or scheduled bucket upload of daily discharge summaries. Validates rows, handles missing categories safely, and ranks the entire cohort.
          </p>
          <div className="border-t border-hairline pt-2 text-[11px] font-mono text-muted space-y-1">
            <div>Endpoint: <code>POST /v1/discharges/upload</code></div>
            <div>Throughput: 1,000 rows in 1.89s</div>
            <div>Status: Production Ready</div>
          </div>
        </div>

        {/* Architecture Mode 3 */}
        <div className="p-5 bg-surface hairline-border rounded-sm space-y-3 relative overflow-hidden opacity-90 border-dashed">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono px-2 py-0.5 bg-caution/15 text-caution rounded font-bold uppercase">
              DESIGN SPECIFICATION ONLY
            </span>
            <Layers className="w-5 h-5 text-muted" />
          </div>
          <h3 className="font-serif text-lg font-semibold text-primary">
            HL7 / FHIR Encounter Adapter
          </h3>
          <p className="text-xs text-muted leading-relaxed">
            SMART-on-FHIR Encounter resource subscription listening to <code>Encounter.status = 'finished'</code> events, normalizing US-Core profiles to model features.
          </p>
          <div className="border-t border-hairline pt-2 text-[11px] font-mono text-muted space-y-1">
            <div>Protocol: HL7 FHIR v4.0.1 Subscription</div>
            <div>Profile: US Core Encounter Profile</div>
            <div>Status: Architecture Design Only</div>
          </div>
        </div>
      </div>

      {/* Code Snippets Section */}
      <div className="bg-surface p-6 rounded-sm hairline-border space-y-4">
        <div className="border-b border-hairline pb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-primary">
            Implementation Examples
          </h2>
          <span className="text-xs text-muted font-mono">HTTP & TypeScript</span>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-mono font-bold text-muted uppercase">
                cURL POST Example
              </span>
              <button
                onClick={() => handleCopy(curlExample, "curl")}
                className="text-xs text-accent hover:underline flex items-center gap-1 font-mono"
              >
                {copiedSnippet === "curl" ? (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="p-3 bg-canvas text-primary hairline-border rounded-sm font-mono text-xs overflow-x-auto">
              {curlExample}
            </pre>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-mono font-bold text-muted uppercase">
                TypeScript SDK Client
              </span>
              <button
                onClick={() => handleCopy(tsExample, "ts")}
                className="text-xs text-accent hover:underline flex items-center gap-1 font-mono"
              >
                {copiedSnippet === "ts" ? (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="p-3 bg-canvas text-primary hairline-border rounded-sm font-mono text-xs overflow-x-auto">
              {tsExample}
            </pre>
          </div>
        </div>
      </div>

      {/* Security, Privacy & Compliance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 bg-surface hairline-border rounded-sm space-y-3">
          <div className="flex items-center gap-2 text-primary font-semibold text-sm">
            <Lock className="w-4 h-4 text-accent" />
            <span>Security & Privacy Guardrails</span>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            ReadmitIQ is architected with strict Zero-PHI boundaries. The model consumes only de-identified clinical covariates (age brackets, diagnostic ICD categories, lab counts).
          </p>
          <ul className="text-xs space-y-2 text-muted list-disc list-inside pt-1">
            <li><strong>No PHI Stored:</strong> No patient names, MRNs, phone numbers, or social security numbers are accepted by the API schema.</li>
            <li><strong>Single Sign-On (SSO):</strong> Production deployment requires SAML 2.0 / OAuth2 OIDC integration with hospital active directory.</li>
            <li><strong>Role-Based Access Control (RBAC):</strong> Separation between Triage Nurse, Discharge Coordinator, Physician, and Auditor roles.</li>
          </ul>
        </div>

        <div className="p-6 bg-surface hairline-border rounded-sm space-y-3">
          <div className="flex items-center gap-2 text-primary font-semibold text-sm">
            <Cpu className="w-4 h-4 text-accent" />
            <span>Deployment Topologies</span>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            ReadmitIQ can be deployed under three distinct enterprise network models depending on institutional compliance requirements:
          </p>
          <ul className="text-xs space-y-2 text-muted list-disc list-inside pt-1">
            <li><strong>On-Premises Edge Container:</strong> Dockerized FastAPI container running inside hospital DMZ behind existing reverse proxies.</li>
            <li><strong>Private VPC Endpoint:</strong> Cloud deployment within customer-isolated AWS/GCP VPC connected via AWS Direct Connect or VPN tunnel.</li>
            <li><strong>Offline Air-Gapped Appliance:</strong> Fully self-contained bundled static bundle and local SQLite/PostgreSQL store with zero outbound internet access.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
