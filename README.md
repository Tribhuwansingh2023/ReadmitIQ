# ReadmitIQ: Capacity-Aware Hospital Readmission Decision Support Platform

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen.svg)](https://lightgbm.readthedocs.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange.svg)](https://github.com/facebookresearch/faiss)
[![Notebooks](https://img.shields.io/badge/Notebooks-6_Stages-purple.svg)](notebooks/stages/)
[![Tests](https://img.shields.io/badge/Tests-145_Passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_RTX_Accelerated-76B900.svg?logo=nvidia&logoColor=white)](notebooks/stages/)

</div>

> **ReadmitIQ** is a clinical decision support system for 30-day all-cause hospital readmission risk prediction among diabetic inpatients. Rather than functioning as a passive predictive dashboard, ReadmitIQ operationalizes risk intelligence into a capacity-aware, evidence-supported discharge coordination workflow: the **ReadmitIQ ActionLoop**.

---

## Table of Contents

- [The ReadmitIQ ActionLoop](#the-readmitiq-actionloop)
- [Signature Differentiators](#signature-differentiators)
- [Scientific Evaluation & Benchmark Results](#scientific-evaluation--benchmark-results)
- [Project Notebooks (6-Stage Pipeline)](#project-notebooks-6-stage-pipeline)
- [Quickstart: Clone and Run](#quickstart-clone-and-run-on-any-laptop)
- [Running the Test Suite](#running-the-automated-test-suite)
- [Repository Structure](#repository-structure)
- [Author & Project Lead](#author--project-lead)
- [Ethical Disclaimer](#ethical--regulatory-disclaimer)
- [License](#license)

---

## The ReadmitIQ ActionLoop

```
+---------------------------------------------------------------------------------------------------+
|                                      READMITIQ ACTIONLOOP                                         |
+---------------------------------------------------------------------------------------------------+

     [ 1. Risk Intelligence ]                       [ 2. Precedent Evidence ]
     - Frozen LightGBM 42-Feature Score             - FAISS Inner-Product Search (k=20)
     - Isotonic Calibrated Probability              - 64-dim Clinical SVD Embedding
     - Top Risk Increasing / Protective SHAP        - Wilson 95% CI & Methodological Disclosure
                 \                                              /
                  \                                            /
                   v                                          v
                              [ 3. Care Action Plan ]
                              - Automated Protocol Tasks Triggered by SHAP Drivers
                              - Status Lifecycle: Pending -> In-Progress -> Completed
                              - Assignment, Due Dates, and Mandatory Completion Notes
                                         |
                                         v
                         [ 4. Capacity-Aware Triage Queue ]
                         - Deterministic Ranking: Raw Score DESC -> Arrival DESC -> ID ASC
                         - Eliminates Isotonic Probability Plateaus
                         - Nurse Outreach Capacity Thresholding (e.g., Top 15 Shifts)
                                         |
                                         v
                         [ 5. Closed-Loop Follow-up ]
                         - Coordinator Touchpoints (Phone, Home Visit, Med Rec)
                         - Unified Chronological Audit Trail & Event History
                         - Sealed Ground-Truth 30-Day Registry Outcome Surveillance
```

1. **Risk Intelligence:** Predicts raw scores and empirical probabilities calibrated via non-parametric isotonic regression. Explains predictions with fast TreeSHAP feature attributions.
2. **Precedent Evidence:** Retrieves k=20 structurally similar historical encounters from a patient-disjoint reference split (n=69,535) via 64-dimensional SVD vector projections. *Methodological disclosure: Precedent retrieval provides clinical case context; it does not add independent predictive ROC-AUC lift.*
3. **Care Transition Action Plan:** Automatically seeds evidence-based discharge tasks derived from patient-specific risk drivers (medication reconciliation, glycemic education, 48h cardiology follow-up).
4. **Capacity-Aware Triage:** Dynamically ranks encounters based on hospital nurse outreach capacity. Enforces strict deterministic tie-breaking (`raw_score DESC`, `received_at DESC`, `id ASC`) to eliminate isotonic calibration ties.
5. **Closed-Loop Follow-up:** Tracks every coordinator touchpoint, audit event, and sealed 30-day post-discharge outcome into a single chronological timeline.

---

## Signature Differentiators

### 1. What-If Risk Sensitivity Explorer
Allows multidisciplinary care teams to simulate how hypothetical variations in clinical factors (such as reducing acute length of stay, optimizing medication regimens, or establishing outpatient follow-up) would shift the model's score.
- **Side-by-side comparison:** Evaluates modified features through the actual, unmocked LightGBM + Isotonic serving pipeline (`POST /v1/what-if`).
- **Delta analysis:** Computes raw score delta, calibrated probability change, relative risk shift percentage, and serving tier transitions.
- **Scientific guardrail:** Explicitly disclosed as associative statistical model sensitivity, *not* a causal treatment effect estimate.

### 2. Trust & Safety Surveillance Monitor
Continuously surveils production pipeline health to protect patient safety:
- **Population Stability Index (PSI):** Real-time monitoring of raw score distribution drift against frozen training deciles (PSI < 0.10 Stable, >= 0.25 High Drift Alert).
- **Feature Covariate Drift:** Tracks covariate shifts across individual predictors against training references.
- **RAG Vector Store Health:** Live health metrics for the FAISS index (n=69,535, 64 dimensions, 0 query errors).
- **Data Quality & Rejection Audit:** Logs all rejected CSV rows, formatting errors, and schema conformity rates in SQLite.
- **Small-n Statistical Power Guardrails:** Withholds live empirical discrimination claims until sample sizes reach n >= 200 resolved outcomes with >= 20 positives. Flags demographic subgroups with n < 100 encounters with an **Unpowered Small-n Guard** to prevent premature compliance assertions.

### 3. Enterprise Clinical Explainability & Trust Engine (XAI v2)
Provides transparent, auditable, and clinically grounded explanations for every individual encounter and the model as a whole:
- **Local TreeSHAP Attribution:** Exact mathematical log-odds decomposition separating risk-increasing vs protective factors with non-causal clinical wording. Max validation violation: 3.60e-7.
- **Telescoping Calibrated Waterfall:** Converts raw log-odds steps into calibrated probability points via order-dependent decomposition that telescopes to the final calibrated risk.
- **Data Completeness Segregation:** Missing and unrecorded observations (`None` lab tests, `Missing` administrative fields) are strictly isolated into a neutral *"Information Not Recorded"* panel.
- **Multi-Method Consensus & Quality Audits:** Verified against LightGBM Gain (rho=0.953) and Permutation Importance (rho=0.885). Passes Adebayo model-randomization sanity checks and demonstrates 3.16x higher faithfulness than random feature ablation.
- **Precedent Agreement & Local Reliability:** Direct consistency checks against k=20 peer readmission Wilson 95% CIs and 10-bin empirical validation readmission calibrations.
- **Full Architecture & Metrics:** See [docs/explainable_ai.md](docs/explainable_ai.md) and [reports/xai/](reports/xai/).

---

## Scientific Evaluation & Benchmark Results

Evaluated on strictly partitioned, patient-disjoint held-out data (N=101,766 encounters, split by `patient_nbr` into 69.5k train / 17.3k val / 14.9k test).

| Metric | Primary Frozen Model | Baseline Comparison | Clinical Note |
| :--- | :--- | :--- | :--- |
| **Validation ROC-AUC** | **0.6731 [0.6602, 0.6863]** | 0.6552 (Logistic Reg.) | 500-sample bootstrap 95% CI |
| **Validation Brier Score** | **0.0894** | 0.2190 (Uncalibrated) | 59.2% reduction via Isotonic Calibration |
| **Paired Delta-AUC (RAG vs Tabular)** | **0.0000** | 95% CI crosses zero | Retrieval provides context, not lift |
| **Blinded Delta-AUC (No Race/Gender)** | **+0.0007** | 0.6738 vs 0.6731 | Demographics omitted without performance penalty |
| **Top 10% Precision (Capacity)** | **23.76%** | 11.37% (Base Prevalence) | **2.09x enrichment** over random outreach |

---

## Project Notebooks (6-Stage Pipeline)

The complete ML pipeline is documented across **6 fully-executed, GPU-accelerated Jupyter notebooks** in [`notebooks/stages/`](notebooks/stages/). Each notebook is self-contained, slide-ready, and produces clean outputs with zero warnings.

| Stage | Notebook | Description | Status |
|:---:|:---|:---|:---:|
| 1 | [stage1.ipynb](notebooks/stages/stage1.ipynb) | Problem Framing, Clinical Economics & System Architecture | Executed |
| 2 | [stage2.ipynb](notebooks/stages/stage2.ipynb) | Data Cleaning, Preprocessing & Patient-Level Splitting | Executed |
| 3 | [stage3.ipynb](notebooks/stages/stage3.ipynb) | RAG Feature Engineering with Vector Retrieval (FAISS) | Executed |
| 4 | [stage4.ipynb](notebooks/stages/stage4.ipynb) | Classical ML Models (Logistic Regression & Random Forest) | Executed |
| 5 | [stage5.ipynb](notebooks/stages/stage5.ipynb) | Transformer-Based Model & Clinical Note Embeddings (GPU) | Executed |
| 6 | [stage6.ipynb](notebooks/stages/stage6.ipynb) | Model Evaluation, SHAP Explainability, Fairness & Drift Audit | Executed |

> **HTML Reports:** Fully rendered, shareable HTML exports of every stage are available in [`reports/stages/`](reports/stages/).

### Running the Notebooks

```bash
# Install notebook dependencies
pip install -r requirements-stages.txt

# Execute a specific stage in-place (clean, no warnings)
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/stages/stage3.ipynb

# Export any stage to a standalone HTML report
jupyter nbconvert --to html notebooks/stages/stage3.ipynb \
  --output stage3.html --output-dir reports/stages/
```

> **GPU Note:** Stages 3 and 5 automatically detect and use an **NVIDIA GPU** (RTX 3050 tested) via CUDA for accelerated embedding and retrieval. A CPU fallback is used if no GPU is detected.

---

## Quickstart: Clone and Run on Any Laptop

### Prerequisites
- **Python:** 3.11 or 3.12 (Python 3.12 recommended)
- **Node.js:** v18, v20, or v22 with `npm`
- **Git:** Standard git client
- **GPU (optional):** NVIDIA CUDA-compatible GPU for accelerated notebook stages

### 1. Clone the Repository
```bash
git clone https://github.com/Tribhuwansingh2023/ReadmitIQ.git
cd ReadmitIQ
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd web
npm install
cd ..
```

### 4. Verify Reproducibility
Run the automated environment and asset verifier:
```bash
python scripts/verify_reproducibility.py
```
*Validates Python runtime, all 7 required frozen model assets in `models/final/`, package dependencies, frontend toolchain, and deterministic inference output.*

### 5. Start the Application

#### Option A: Using Platform Launch Scripts
- **Windows:**
  - Launch Backend: double-click `scripts\run_backend.bat`
  - Launch Frontend: double-click `scripts\run_frontend.bat`
- **macOS / Linux:**
  - `chmod +x scripts/*.sh`
  - Backend: `./scripts/run_backend.sh`
  - Frontend: `./scripts/run_frontend.sh`

#### Option B: Using Direct Commands
```bash
# Terminal 1 - Backend:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend:
cd web && npm run dev
```

#### Option C: Using Docker Compose
```bash
docker-compose up --build
```

Access the application:
- **Web Command Center:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Reference:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## Running the Automated Test Suite

ReadmitIQ includes **145 automated tests** with zero external network dependencies:
```bash
# Run all 145 tests
pytest

# Run ActionLoop integration tests specifically
pytest tests/test_actionloop.py -v

# Run XAI explainability tests
pytest tests/test_xai_explainability.py -v

# Run frontend type-check & production build
cd web && npm run build
```

---

## Repository Structure

```
ReadmitIQ/
├── api/                        # FastAPI REST service & schema validation
│   ├── main.py                 # Core app gateway & middleware
│   ├── v1.py                   # ActionLoop v1 router (18 endpoints)
│   ├── service.py              # ModelService inference orchestrator
│   ├── database.py             # SQLite WAL persistence & audit logger
│   ├── schemas.py              # Pydantic clinical feature contracts
│   └── feed.py                 # Simulation engine & Time Machine
├── models/
│   └── final/                  # Frozen runtime model assets (tracked)
│       ├── model.joblib         # Frozen LightGBM model (42 features)
│       ├── calibrator.joblib    # Isotonic probability calibrator
│       ├── preprocessor.joblib  # Feature transformer pipeline
│       ├── structured_svd.joblib# 64-dim latent vector projection
│       ├── retriever_faiss.index# FAISS IndexFlatIP vector store (17.8 MB)
│       └── retriever_meta.parquet # 69.5k disjoint reference cohort
├── notebooks/                  # Jupyter notebooks (research & pipeline)
│   ├── stages/                 # 6-stage fully-executed pipeline notebooks
│   │   ├── stage1.ipynb        # Problem Framing & Architecture
│   │   ├── stage2.ipynb        # Data Preprocessing & Splitting
│   │   ├── stage3.ipynb        # RAG / Vector Retrieval Engineering
│   │   ├── stage4.ipynb        # Classical ML Models
│   │   ├── stage5.ipynb        # Transformer & Note Embeddings (GPU)
│   │   ├── stage6.ipynb        # Evaluation, SHAP, Fairness & Drift
│   │   └── _stage_helpers.py   # Shared utilities (GPU detection, badges)
│   ├── 01_eda_and_data_preparation.ipynb  # EDA notebook
│   ├── colab_embed_notes.ipynb # Google Colab: note embedding pipeline
│   └── xai.ipynb               # XAI deep-dive notebook
├── web/                        # React 19 + TypeScript + Vite frontend
│   ├── src/components/         # ActionPlanBoard, PrecedentOrbit, WhatIfExplorer, etc.
│   ├── src/pages/              # DeskPage, DischargeBriefModal, MonitoringPage
│   └── src/api/client.ts       # Typed API client
├── reports/                    # Generated outputs
│   ├── stages/                 # HTML exports of all 6 stage notebooks
│   │   ├── stage1.html
│   │   ├── stage2.html
│   │   ├── stage3.html
│   │   ├── stage4.html
│   │   ├── stage5.html
│   │   └── stage6.html
│   └── xai/                    # XAI figures & evaluation reports
├── docs/                       # Comprehensive documentation suite
│   ├── architecture.md         # System architecture & component data flows
│   ├── actionloop_specification.md # Detailed ActionLoop Stages 1-5 spec
│   ├── differentiators.md      # What-If Explorer & Trust & Safety deep-dive
│   ├── portability_guide.md    # Multi-platform setup & clone runbook
│   ├── api_reference.md        # Full REST API endpoint reference
│   ├── clinical_guardrails.md  # Research limitations & methodological boundaries
│   ├── evaluation_and_scientific_results.md # Frozen validation benchmarks
│   ├── fairness_and_subgroup_audit.md # Demographic parity & small-n power
│   ├── data_dictionary.md      # The 42 clinical features & mappings
│   ├── rag_retrieval_architecture.md # FAISS latent space & SVD design
│   ├── monitoring_and_drift_surveillance.md # PSI math & surveillance rules
│   └── troubleshooting.md      # Common setup issues & solutions
├── scripts/                    # Launchers & verification utilities
│   ├── verify_reproducibility.py # System verification CLI
│   ├── run_backend.bat / .sh   # Cross-platform backend launchers
│   ├── run_frontend.bat / .sh  # Cross-platform frontend launchers
│   ├── check_system_health.py  # Health check utility
│   └── export_demo_encounters.py # Demo CSV generator
├── tests/                      # Automated test suite (145 passing tests)
│   ├── test_actionloop.py      # ActionLoop end-to-end integration tests
│   ├── test_v1_workflow.py     # Worklist, tasks, and audit trail tests
│   ├── test_xai_explainability.py # XAI & SHAP decomposition tests
│   ├── test_audit_leakage.py   # Zero patient leakage audit
│   └── test_rag_leakage.py     # RAG split isolation verification
├── data/
│   └── processed/              # Cleaned splits & embeddings (.npy, .parquet)
├── scratch/                    # Development & debugging utility scripts
├── Dockerfile.backend          # Production backend containerfile
├── Dockerfile.frontend         # Multi-stage frontend containerfile
├── docker-compose.yml          # Container orchestration configuration
├── requirements.txt            # Python dependencies (API & backend)
├── requirements-stages.txt     # Python dependencies (notebooks pipeline)
├── pyproject.toml              # Toolchain configuration
└── .env.example                # Safe environment configuration template
```

---

## Author & Project Lead

**Tribhuwan Singh**
- **GitHub:** [@Tribhuwansingh2023](https://github.com/Tribhuwansingh2023)
- **Repository:** [Tribhuwansingh2023/ReadmitIQ](https://github.com/Tribhuwansingh2023/ReadmitIQ)
- **Contact:** [webosingh93@gmail.com](mailto:webosingh93@gmail.com)

---

## Ethical & Regulatory Disclaimer

ReadmitIQ is an investigational research prototype designed to demonstrate capacity-aware discharge coordination and transparent model surveillance. It has **not** received FDA 510(k), De Novo clearance, or CE mark approval as a medical device (SaMD). It is **not** intended to replace clinical judgment. All clinical decisions regarding discharge readiness, patient outreach, and treatment interventions must be made by qualified healthcare professionals.

---

## License

Distributed under the [MIT License](LICENSE).
