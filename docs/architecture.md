# ReadmitIQ System Architecture

ReadmitIQ is an end-to-end clinical decision support platform for 30-day hospital readmission risk prediction among diabetic inpatients.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Data["Data & Preprocessing Layer"]
        Raw["UCI Diabetic Encounters (101,766 encounters / 71,518 patients)"]
        Clean["Data Cleaning & Preprocessing (99,340 eligible encounters)"]
        Split["Patient-Disjoint Split (70% Train / 15% Val / 15% Test)"]
        Raw --> Clean --> Split
    end

    subgraph FeaturePipeline["Feature Engineering & RAG Pipeline"]
        FE["Base Feature Engineering (41 clean tabular features)"]
        SVD["64-dim Truncated SVD (Fit strictly on Train features)"]
        FAISS["FAISS Cosine Similarity Index (69,535 Train encounters)"]
        RAG["Structured RAG Retriever (k=20 Precedents)"]
        OOF["5-Fold Patient-Disjoint OOF (Train RAG feature generation)"]
        
        Split --> FE
        FE --> SVD --> FAISS
        FE --> OOF --> RAG
    end

    subgraph Modeling["Modeling & Serving Layer"]
        Model["LightGBM Classifier (CV-tuned depth=4, lr=0.05)"]
        Calib["Isotonic Calibrator (Fit strictly on Train 5-Fold OOF)"]
        SHAP["TreeExplainer Attribution Engine"]
        Freeze["models/FROZEN_SELECTION.json"]

        RAG --> Model
        Model --> Calib
        Model --> SHAP
        Calib --> Freeze
    end

    subgraph Application["Application Layer"]
        API["FastAPI REST Backend (:8000)"]
        Web["React + TypeScript + Tailwind SPA (:5173)"]
        
        Freeze --> API
        API <--> Web
    end
```

---

## 2. Structured RAG Retrieval Mechanism

The Structured RAG engine provides explainable case-based clinical reasoning by identifying the historical patients in the training set who most closely resemble the current patient:

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Care Team / Clinician
    participant UI as ReadmitIQ Web SPA
    participant API as FastAPI Backend
    participant Pipe as Preprocessor Base
    participant SVD as 64-dim Truncated SVD
    participant FAISS as FAISS Index (Train Split)
    participant Model as Calibrated LightGBM

    Clinician->>UI: Enter patient discharge parameters
    UI->>API: POST /predict {patient_features}
    API->>Pipe: Transform 41 base attributes to 141-dim vector
    Pipe->>SVD: Project to 64-dim normalized latent vector
    SVD->>FAISS: Search top-20 nearest neighbors (L2-normalized Inner Product)
    FAISS-->>API: Return neighbor indices, similarities, and historical 30d outcomes
    API->>API: Compute neighborhood readmission rate (rag_struct_rate_k20)
    API->>Model: Predict raw probability using 42 features
    Model->>API: Isotonic calibration -> Calibrated Risk Probability
    API->>API: Assign Risk Tier (Critical, High, Medium, Low) & SHAP attributions
    API-->>UI: Return risk probability, tier, plain-language factors & similar patients
    UI-->>Clinician: Display calibrated risk gauge, SHAP cards & historical precedents
```

---

## 3. Retrospective Cohort Replay Architecture

To simulate operational clinical workload allocation, the Cohort Command Center performs real-time capacity triage replay on the held-out Test split (N = 14,903):

- **Capacity Fraction ($c$):** Adjustable from 1% to 30%.
- **Flagged Volume ($N_c$):** $N_c = \text{round}(c \times 14,903)$.
- **True Readmissions Caught ($R_c$):** Number of true $<30$-day readmissions in top $N_c$ ranked patients.
- **Precision at Capacity:** $P_c = R_c / N_c$.
- **Lift vs Random Prevalence:** $\text{Lift} = P_c / 0.1137$.
