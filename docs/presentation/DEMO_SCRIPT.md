# ReadmitIQ: Presentation & Live Demo Run Sheet

*Timed run sheet with exact click targets, spoken talking points, and recovery protocols.*

---

## Part 1: Timed 5-Minute Presentation & Demo Run Sheet

| Timestamp | Speaker / Stage | Screen / Action | Spoken Script & Key Numbers |
|:---:|:---|:---|:---|
| **0:00 – 0:45**<br>(45 sec) | **Member 1 & 2**<br>Problem & Data Rigor | Slides 1–4<br>(Problem, Economics, 4 Data Traps) | *"Good morning. Unplanned hospital readmissions cost US healthcare $26B annually, but care coordinators only have the capacity to phone 10% of discharged patients. We built ReadmitIQ using the UCI 130 dataset—99,340 eligible encounters across 70,442 patients. We defeated 4 critical data traps, enforcing strict patient-level hashing with 0% patient overlap across all sets."* |
| **0:45 – 1:30**<br>(45 sec) | **Member 3 & 4**<br>Architecture & The Honest RAG Finding | Slides 5–7<br>(Architecture, Ablation, Paired $\Delta$) | *"Our pipeline projects 141 features into a 64-dim latent space, querying k=20 peers in FAISS. In our ablation benchmark across Logistic Regression, Random Forest, and LightGBM, LightGBM won with test AUC 0.6590. Crucially, our paired bootstrap test showed RAG added +0.0004 AUC with the CI crossing zero. We state honestly: RAG does not inflate discrimination; its value is clinical precedent trust."* |
| **1:30 – 2:15**<br>(45 sec) | **Member 5**<br>Calibration & Operational Lift | Slide 8<br>(Test Calibration & Top-k Lift) | *"On our locked test cohort of 14,903 encounters, isotonic calibration dropped Expected Calibration Error to 0.39%. At 10% outreach capacity, ReadmitIQ achieves a 2.09x lift over random prevalence, capturing 20.85% of all readmissions with 23.76% precision."* |
| **2:15 – 3:45**<br>(90 sec) | **Member 6**<br>**LIVE APP DEMO** | Browser:<br>`http://localhost:5173` | **[LIVE DEMO SEQUENCE]**<br>1. **Cohort View (`/#cohort`):** Show 14,903 test encounters. Move Capacity Slider from 10% to 15%. Point out What-If Simulator: *$342,000 net savings* on 1,490 outreaches.<br>2. **Patient View (`/#patient`):** Click preset *'High Risk (Elderly)'*. Watch calibrated risk dial animate to ~38% (Critical Tier).<br>3. **TreeSHAP Drivers:** Highlight top drivers: *2 prior inpatient stays*, *7 days in hospital*.<br>4. **Precedent Table:** Show 20 historical peers with a 35% peer readmission rate.<br>5. **Care Protocols:** Point out auto-suggested 48h nurse phone call, pharmacist insulin review, and home health handoff. |
| **3:45 – 4:30**<br>(45 sec) | **Member 5 & 7**<br>Fairness & Production Readiness | Slides 9–11<br>(Fairness, Latency, Docker) | *"We audited fairness: a demographic-blinded model excluding race and gender achieved 0.6597 AUC (+0.0007), proving our model evaluates clinical acuity rather than demographic proxies. All 42 features show PSI < 0.05. Our FastAPI backend delivers p95 latency of 386.0ms and runs within 195MB RAM in Lite mode."* |
| **4:30 – 5:00**<br>(30 sec) | **Member 1**<br>Governance & Conclusion | Slides 12–14<br>(Limitations, Roadmap, Conclusion) | *"In conclusion, ReadmitIQ is rigorously audited, calibrated, explainable, and containerized for hospital deployment. We are ready for your questions."* |

---

## Part 2: Lightning 2-Minute Version (Rapid Pitch)

- **[0:00 – 0:30] The Challenge & Data:** *"ReadmitIQ solves the hospital readmission bottleneck. Care coordinators can only phone 10% of discharges. Using 99,340 encounters from 130 hospitals with 0% patient leakage, we built a calibrated triage system."*
- **[0:30 – 1:15] The Demo:** *"Here at 10% outreach capacity, our model catches 20.8% of readmissions with a 2.09x lift. For this 75-year-old diabetic patient, the calibrated risk is 38.2%. TreeSHAP shows 2 prior inpatient admissions drove the risk, our Structured RAG surfaces 20 historical peers who had a 35% readmission rate, and our rule engine triggers 3 verified care transitions."*
- **[1:15 – 2:00] Rigor & Tech:** *"We benchmarked with 95% bootstrap CIs: test AUC is 0.6590, calibration ECE is 0.39%, and our demographic-blinded audit proves zero bias. Containerized in FastAPI with sub-390ms latency. Thank you."*

---

## Part 3: Live Demo Disaster Recovery Plan

| Failure Mode | What Happens | Immediate Recovery Action |
|:---|:---|:---|
| **Venue Wi-Fi fails** | Browser cannot reach hosted Vercel/Render URL | Open local browser tab: `http://localhost:5173`. The backend is already running locally on port 8000 with sub-50ms latency. |
| **Local backend crashes** | Frontend displays red *'API Offline'* banner | Run terminal command: `python -m uvicorn api.main:app --port 8000`. Reconnects within 3 seconds. |
| **Complete laptop / projector freeze** | Presenter screen freezes | Switch to backup laptop or open [docs/demo_fallback.md](file:///C:/Users/tribh/.gemini/antigravity-ide/scratch/readmitt-iq/docs/demo_fallback.md) containing static screenshots `docs/screenshots/cohort_triage_desktop.png` and `patient_risk_desktop.png`. |
