# ReadmitIQ: 7-Member Presentation Speaking Plan

*A synchronized 5-minute talk division for 7 members with exact handover lines and ownership boundaries.*

---

## Overview
- **Total Presentation Time:** 5 Minutes (300 Seconds) + Q&A
- **Structure:** 7 Handover Segments (~40–45 seconds each)
- **Visual Presence:** Seamless speaker transitions to showcase collaborative excellence to the Cognizant judging panel.

---

## Speaker Roster & Handover Cues

### Member 1: Clinical Problem & Healthcare Economics (0:00 – 0:40)
- **Role Title:** Clinical Domain Lead & Product Formulation
- **Slides:** Slide 1 (Title) & Slide 2 (The $26B Readmission Bottleneck)
- **Core Message:** The $26B annual readmission penalty under Medicare HRRP and the care-coordinator bandwidth bottleneck (10% capacity).
- **Handover Line:** *"To explain how we built an uncompromised data foundation for this clinical challenge, I'll pass to Member 2."*

---

### Member 2: Data Hygiene & The 4 Leakage Traps (0:40 – 1:20)
- **Role Title:** Data Engineering & Leakage Isolation
- **Slides:** Slide 3 (Cohort Findings) & Slide 4 (4 Data Traps)
- **Core Message:** 99,340 eligible encounters across 70,442 patients. Grouped patient hashing ensuring **0% patient overlap**, mortality exclusions, and informative lab missingness.
- **Handover Line:** *"With a leak-free data split established, Member 3 will walk through our end-to-end architecture and model benchmarking."*

---

### Member 3: Architecture & Model Leaderboard (1:20 – 2:00)
- **Role Title:** ML Modeling & Calibration Lead
- **Slides:** Slide 5 (Architecture) & Slide 7 (Ablation Leaderboard)
- **Core Message:** 141-feature tabular pipeline, LightGBM model selection (Val AUC 0.6731), and bootstrap 95% confidence intervals.
- **Handover Line:** *"Now, Member 4 will reveal our Structured RAG engine and our key scientific finding regarding precedent retrieval."*

---

### Member 4: Structured RAG & The Honest Finding (2:00 – 2:40)
- **Role Title:** Retrieval-Augmented Generation Specialist
- **Slides:** Slide 6 (Structured RAG Design) & Slide 7 (The Honest Finding Callout)
- **Core Message:** 64-dim latent SVD projection, FAISS k=20 retrieval, and the honest scientific callout: paired test delta AUC is +0.0004 (CI crosses zero). RAG provides clinical trust, not metric inflation.
- **Handover Line:** *"Next, Member 5 will share how our isotonic calibration translates this into operational hospital lift and algorithmic fairness."*

---

### Member 5: Calibration, Operational Lift & Fairness (2:40 – 3:20)
- **Role Title:** Explainability, Calibration & Fairness Auditor
- **Slides:** Slide 8 (Held-Out Test Calibration) & Slide 9 (Fairness & SHAP)
- **Core Message:** Test AUC 0.6590, 2.09x lift at 10% capacity, ECE 0.39%, and the demographic-blinded model audit proving zero demographic bias (Blinded AUC 0.6597).
- **Handover Line:** *"Now, Member 6 will take us live into the ReadmitIQ Care-Team Command Center."*

---

### Member 6: Live Product Demonstration (3:20 – 4:20)
- **Role Title:** Full-Stack & Clinical UX Lead
- **Slides:** Live Application (`http://localhost:5173` or Slide 10 UI Screens)
- **Core Message:** Cohort Triage capacity slider, What-If ROI impact simulator ($342k net savings), Bedside Patient Assessment dial, TreeSHAP drivers, and peer precedent inspection.
- **Handover Line:** *"Finally, Member 7 will detail our production serving architecture, latency benchmarks, and enterprise readiness."*

---

### Member 7: Production DevOps, Latency & Governance (4:20 – 5:00)
- **Role Title:** DevOps, API & Production Infrastructure Lead
- **Slides:** Slide 11 (Production Readiness), Slide 12 (Limitations), Slide 14 (Next Steps)
- **Core Message:** FastAPI backend, sub-390ms p95 latency, dual Lite serving mode (<2MB, <195MB RAM), PSI drift telemetry, and HL7 FHIR roadmap.
- **Closing Line:** *"ReadmitIQ is audited, calibrated, explainable, and production-ready. We welcome the panel's questions."*
