# ReadmitIQ Web Interface Walkthrough

ReadmitIQ features an editorial-grade, accessible web interface built with React 19, TypeScript, and Vite.

---

## 1. Discharge Desk (`/desk`)
The operational cockpit for clinical care coordinators and discharge planning teams.

- **Headline Odometers:** Displays live counts of encounters awaiting triage, open High/Critical cases, resolved 30-day outcomes, and queue throughput.
- **Capacity Follow-up Control:** A dedicated capacity slider allows coordinators to restrict the active worklist to the top $N$ patients (e.g. top 15) matching nurse staffing levels.
- **Deterministic Priority Queue:** Each row displays its deterministic queue position (`#1`, `#2`, etc.), external encounter reference, serving risk tier glyph, calibrated readmission probability, and raw LightGBM score.
- **Permanent Simulation Disclosure:** When the live historical replay feed is active, a status banner informs clinicians that encounters are drawn from held-out evaluation splits with outcomes sealed until Time Machine release.
- **Keyboard Navigation:** Clinicians can navigate the worklist rapidly using `j` (down), `k` (up), and `Enter` (open patient decision brief).

---

## 2. Encounter Decision Brief Modal
Clicking any patient opens the comprehensive decision brief structured along the 5 ActionLoop stages:

1. **Stage 1 (Risk Intelligence):**
   - High-contrast SVG Calibrated Risk Dial displaying the 30-day readmission probability.
   - Raw model score and serving tier badge.
   - Clinical workflow status stepper (`New` $\rightarrow$ `Assigned` $\rightarrow$ `Contacted` $\rightarrow$ `Scheduled` $\rightarrow$ `Closed`).
   - Human-in-the-loop Clinician Risk Override with mandatory documented rationale.
   - Fast TreeSHAP waterfall cards breaking down top risk-increasing and top protective factors.
2. **Stage 2 (Precedent Evidence & Precedent Orbit):**
   - Interactive radial orbit visualizer plotting the current patient at the center with 20 historical peers positioned radially by geometric cosine distance ($d = 1 - \cos\theta$).
   - Neighborhood readmission rate with Wilson 95% confidence intervals.
   - Prominent scientific guardrail disclosure explaining that peer retrieval provides clinical context, not incremental predictive lift.
3. **Stage 3 (Care Transition Action Plan):**
   - Automated protocol tasks triggered by patient-specific risk drivers.
   - Task cards with priority badges, assignment dropdowns, and status transitions (`pending`, `in_progress`, `completed`).
   - Custom task addition modal.
4. **Stage 4 (Capacity Triage Context):**
   - Exact queue position within the active backlog.
   - Capacity allocation status ("Within Outreach Capacity" vs "Beyond Outreach Threshold").
   - Deterministic tie-breaker explanation.
5. **Stage 5 (Closed-Loop Follow-up & Audit Timeline):**
   - Touchpoint logger for recording phone calls, home visits, and medication reviews.
   - Reverse-chronological timeline combining intake, outreach actions, task completions, and sealed post-discharge outcomes.

---

## 3. What-If Risk Explorer Tab
- Embedded directly within the patient brief.
- Provides interactive sliders for acute length of stay, prior inpatient admissions, total medications, and emergency room visits.
- Displays immediate side-by-side comparison of baseline vs modified predictions with absolute and relative delta percentage indicators.
- Causal disclaimer reminding care teams that simulations reflect associative statistical responses, not prospective clinical guarantees.

---

## 4. Trust & Safety Surveillance Monitor (`/monitoring`)
- **Headline Stat Cards:** Discharges scored, open backlog, resolved outcomes, and raw score PSI.
- **Risk Tier Allocation vs Expected:** Comparative bar chart and accessible data table contrasting live tier shares against frozen training deciles.
- **Live Empirical ROC-AUC:** Calculates live discrimination and Brier score against released outcomes, with small-n statistical guardrail messages displayed when $n < 200$.
- **RAG Vector Store Health:** Surveillance card reporting FAISS index status, 64-d SVD dimensions, 69k vectors, and logged retrieval errors.
- **Data Quality & Rejection Log:** Audit table listing schema validation failures, rejected CSV rows, and conformity rates.
- **Subgroup Fairness Audit:** Detailed demographic breakdown by race and age bracket, flagging unpowered slices with an explicit small-n warning.
- **Interactive Time Machine:** Controls to advance the simulated calendar clock by $+7\text{d}$ or $+30\text{d}$ to simulate post-discharge outcome resolution.
