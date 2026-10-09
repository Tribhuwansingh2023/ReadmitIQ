# ReadmitIQ: Care-Team Hospital Readmission Command Center

[![CI Pipeline](https://github.com/readmitt-iq/readmit-iq/actions/workflows/ci.yml/badge.svg)](https://github.com/readmitt-iq/readmit-iq/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?logo=typescript)](https://www.typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg?logo=tailwind-css)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**ReadmitIQ** is an end-to-end clinical decision support platform for predicting 30-day hospital readmission risk among diabetic inpatients at discharge. It bridges the gap between predictive modeling and clinical operations through **calibrated capacity-based risk tiers**, **SHAP feature attribution**, and **Structured Retrieval-Augmented Generation (RAG)** for explainable precedent case retrieval.

---

## Key Highlights

1. **Patient-Level Leakage Prevention:** 99,340 eligible encounters split by `patient_nbr` (70% train / 15% val / 15% test) ensuring **0% patient overlap** across splits.
2. **Structured RAG Engine:** Fast FAISS Inner Product search ($k=20$) across 69,535 historical training cases via 64-dimensional SVD latent projections.
3. **Probability Calibration:** Isotonic regression fit on 5-fold Out-Of-Fold Train predictions reduces Test Brier score by 55.6% (0.219 -> **0.0973**) with Expected Calibration Error (ECE) of **0.0039**.
4. **Honest Scientific Evaluation:** Test set evaluated exactly once under `reports/metrics/test_eval.lock`. Paired bootstrap Delta AUC (+0.0004, 95% CI: [-0.0025, +0.0031]) honestly documents that RAG does not offer statistically significant ranking gains over 41 tabular features, but provides essential case-based clinical explainability.
5. **Operational Care Triage:** Retrospective cohort replay with adjustable capacity sliders (1% to 30%). At 10% capacity, achieves **23.76% precision** (**2.09x lift** over random 11.37% prevalence).
6. **Production Full-Stack:** FastAPI REST backend (<15ms inference latency) + React/TypeScript/Vite/Tailwind command center with zero mock data.

---

## Project Structure

```
readmitt-iq/
├── api/                        # FastAPI REST service & Pydantic schemas
│   ├── main.py                 # API router endpoints
│   ├── schemas.py              # Input/output schemas with derived clinical features
│   └── service.py              # ModelService orchestrator
├── configs/
│   └── config.yaml             # Paths, random seeds, and hyperparameters
├── data/
│   ├── raw/                    # diabetic_data.csv (read-only, UCI Diabetes 130)
│   └── processed/              # train, val, test parquet files
├── docs/                       # Project documentation, cards, and presentation slides
│   ├── architecture.md         # Mermaid diagrams of system dataflow
│   ├── audit_report.md         # Comprehensive leakage and consistency audit
│   ├── data_card.md            # Dataset provenance and limitations
│   ├── model_card.md           # Model card with fairness and bootstrap CIs
│   ├── demo_script.md          # 5-minute live judging demo walkthrough
│   ├── slide_content.md        # Pitch presentation script
│   └── DEPLOY_MANUAL_STEPS.md  # Production cloud deployment guide
├── models/
│   ├── FROZEN_SELECTION.json   # Winning model selection metadata
│   └── final/                  # Serialized model, calibrator, preprocessors, FAISS index
├── notebooks/
│   ├── 01_eda_and_data_preparation.ipynb # Fully executed clean EDA notebook
│   └── colab_transformer_training.ipynb  # GPU DistilBERT fine-tuning notebook
├── reports/
│   ├── figures/                # ROC, PR, Calibration, and SHAP summary plots
│   └── metrics/                # Ablation tables, test results, fairness, and drift JSONs
├── src/                        # Core Python ML library
│   ├── data.py                 # Preprocessing and patient-level splitting
│   ├── features.py             # Feature engineering and rare-category handling
│   ├── rag.py                  # Structured FAISS retriever and OOF generator
│   ├── train.py                # Model training, hyperparameter tuning, calibration
│   ├── evaluate.py             # Bootstrap metrics and paired Delta AUC
│   ├── explain.py              # TreeExplainer SHAP and plain-language formatting
│   └── drift.py                # Population Stability Index (PSI) monitoring
├── tests/                      # Automated test suite (24 tests passing)
├── web/                        # React 19 + TypeScript + Vite + Tailwind SPA
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Local multi-service orchestration
└── requirements.txt            # Pinned Python dependencies
```

---

## Quickstart: Running Locally

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Node.js 18+ and npm 9+

### 1. Backend Setup & Run
```bash
# Clone repository
git clone https://github.com/readmitt-iq/readmit-iq.git
cd readmitt-iq

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server (daemon or foreground)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

### 2. Frontend Setup & Run
```bash
# Open new terminal in web directory
cd web

# Install frontend dependencies
npm install

# Start Vite dev server
npm run dev
```
Web app available at: `http://localhost:5173`

---

## Reproducing the Pipeline from Scratch

To re-run the full pipeline from raw data to frozen models:

```bash
# 1. Clean data and generate patient-disjoint splits
python src/data.py

# 2. Build preprocessors and generate 5-fold OOF Structured RAG features
python src/rag.py

# 3. Train models, perform 5-fold CV tuning, and evaluate validation ablation
python src/train.py

# 4. Freeze winning model (creates models/FROZEN_SELECTION.json & models/final/)
python src/freeze.py

# 5. Execute locked held-out test evaluation (once only)
python src/final_test_eval.py

# 6. Run automated test suite
pytest tests/ -v
```

---

## Transformer (DistilBERT) Colab Workflow

DistilBERT is evaluated as an offline, validation-only benchmark comparison on clinical notes using free-tier GPU execution in Google Colab:
1. Open `notebooks/colab_transformer_training.ipynb` in [Google Colab](https://colab.research.google.com).
2. Connect to a free T4 GPU runtime (**Runtime -> Change runtime type -> T4 GPU**).
3. Upload files from `data/processed/colab_upload/`: `colab_train.parquet` (5.6 MB), `colab_val.parquet` (1.2 MB), and `MANIFEST.json`.
4. Run all cells top to bottom (trains Variant A: Notes Only and Variant B: Notes + RAG for 3 epochs with AdamW and mixed precision).
5. Download exported `colab_outputs.zip` and extract into `reports/colab/`.
6. Run local import and documentation sync:
   ```bash
   python -m src.transformer_import
   python scripts/sync_transformer_docs.py
   ```

<!-- TRANSFORMER:START -->
*Transformer Benchmark Status:* Pending Google Colab T4 GPU execution (`notebooks/colab_transformer_training.ipynb`). The production model was frozen before the transformer results existed.
<!-- TRANSFORMER:END -->

---

## Verification & Test Suite

All 24 automated tests pass:
```bash
pytest tests/ -v
```
Test suite verifies:
- Zero patient overlap across splits (`test_data_split.py`)
- Zero label leakage in notes and tabular features (`test_rag_leakage.py`)
- FAISS index size, metadata alignment, and OOF cross-validation integrity (`test_rag_leakage.py`)
- Test evaluation lock enforcement (`test_stage2_phase4.py`)
- PSI population stability index calculation (`test_stage2_phase4.py`)
- FastAPI endpoints, validation, and error handling (`test_api.py`)

---

## Integration Options

ReadmitIQ supports flexible clinical hospital data integration pathways:

### 1. REST Push (`POST /v1/discharges`) — [Implemented]
- Real-time synchronous HTTP POST from hospital EHR or ADT (Admission-Discharge-Transfer) engines upon patient discharge.
- Accepts single encounter or vectorized batches (up to 5,000 records).
- Sub-2s vectorized scoring throughput with immediate ranked risk tier assignment.

### 2. CSV Batch Drop (`POST /v1/discharges/upload`) — [Implemented]
- Daily or shift-based batch upload by hospital informatics or care coordination teams.
- Standard downloadable template (`GET /v1/templates/discharge.csv`) with full validation reporting (accepted rows vs. per-row error breakdown).

### 3. FHIR / HL7 v2 Adapter — [Design Only — Not Implemented]
- **Architecture Specification:** Inbound HL7 v2 `ADT^A03` (Discharge) or FHIR R4 `Encounter` resource subscription trigger.
- **Resource Mapping:**
  - `Encounter.period.end` -> `received_at`
  - `Encounter.length` -> `time_in_hospital`
  - `Condition` resources where `category=encounter-diagnosis` -> `diag_1_group`, `diag_2_group`, `diag_3_group`
  - `MedicationRequest` resources -> medication usage and dose change counts
  - `Observation` where `code=4548-4` (HbA1c) -> `A1Cresult`
- **Output:** Transformed into ReadmitIQ `PatientEncounterRequest` schema and ingested via internal REST pipeline. Currently specified as architecture design pattern for enterprise integration.

