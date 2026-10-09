# ReadmitIQ: 3-5 Minute Demo Video Storyboard

*Scene-by-scene script and technical production guidelines for the recorded video demonstration.*

---

## Video Specifications
- **Target Duration:** 3 minutes 45 seconds (Max 5 minutes)
- **Resolution:** 1080p (1920x1080, 16:9), 60 fps
- **Audio:** Clear vocal narration, optional subtle ambient background track at -22 dB
- **Source Assets:** Real UI recorded from `http://localhost:5173`, high-res figures from `reports/figures/`, and screenshots in `docs/screenshots/`

---

## Scene Breakdown

### Scene 1: The Clinical Reality (0:00 – 0:45)
- **Visual:** Title card followed by `docs/screenshots/cohort_triage_desktop.png`.
- **On-Screen Text:** *"30-Day Hospital Readmissions: A $26B Healthcare Challenge"*.
- **Narration:**
  > *"Every year, unplanned 30-day hospital readmissions cost US healthcare over $26 billion. Under Medicare's HRRP, hospitals face steep penalties for excess readmissions.
  > But care coordination teams have a bandwidth problem: they can only follow up with 10% of discharged patients.
  > This is ReadmitIQ: an enterprise clinical command center that pairs calibrated machine learning with structured precedent retrieval."*

### Scene 2: Cohort Triage & What-If ROI Simulation (0:45 – 1:30)
- **Visual:** Screen capture of active interaction with the Cohort Triage page. Mouse drags capacity slider from 10% to 15%. Camera zooms into the What-If ROI simulation card.
- **On-Screen Callout:** *"2.09x Lift Over Random | 354 True Readmissions Caught at 10% Capacity"*.
- **Narration:**
  > *"On our held-out test cohort of 14,903 encounters, ReadmitIQ prioritizes patients by operational capacity.
  > At 10% outreach capacity, the care team flags 1,490 high-risk patients, achieving a 2.09x lift over random triage and capturing nearly 24% of all 30-day readmissions.
  > In our What-If Impact Simulator, hospital leadership can model outreach staffing. With an 18% assumed relative risk reduction, this single cohort represents over $340,000 in projected net savings."*

### Scene 3: Bedside Assessment & Structured RAG (1:30 – 2:30)
- **Visual:** Transition to Patient Assessment page. Click preset *'High Risk (Elderly)'*. Risk dial animates to ~38%. Scroll through SHAP factors and the peer precedent table.
- **On-Screen Callout:** *"Calibrated Risk: 38.2% | TreeSHAP Attributions | 20 Historical Peers"*.
- **Narration:**
  > *"At bedside discharge, clinicians don't trust unexplained numbers.
  > For this 75-year-old patient with congestive heart failure and diabetes, our isotonic-calibrated model assigns a 38.2% risk—placing them in the Critical tier.
  > Below, TreeSHAP translates mathematical attributions into plain clinical drivers: two prior inpatient hospitalizations and a 7-day length of stay.
  > Next, our Structured RAG engine searches 69,000 historical training encounters using a 64-dimensional clinical embedding space, surfacing 20 historical peers who experienced a 35% readmission rate.
  > Finally, our deterministic protocol engine triggers three actionable transitions: a 48-hour follow-up call, medication reconciliation, and confirmation of a home health visit."*

### Scene 4: Model Lab & Scientific Honesty (2:30 – 3:15)
- **Visual:** Transition to Model Lab page (`/#lab`). Highlight Ablation Leaderboard and Calibration Curve.
- **On-Screen Callout:** *"LightGBM + RAG: AUC 0.6590 | Paired \Delta AUC: +0.0004 (CI Crosses 0) | ECE: 0.39%"*.
- **Narration:**
  > *"In our Model Lab, we benchmarked Logistic Regression, Random Forest, and LightGBM with and without RAG. LightGBM won with a test AUC of 0.6590.
  > But as responsible engineers, we report our findings honestly: our 1,000-sample paired bootstrap test showed RAG added +0.0004 AUC, with the confidence interval crossing zero. RAG does not inflate discrimination on tabular EHR data; its true value is clinical interpretability and peer case inspection.
  > Furthermore, our Isotonic Calibration dropped Expected Calibration Error to 0.39%, ensuring predicted probabilities match empirical reality."*

### Scene 5: Equity, Production & Wrap-Up (3:15 – 3:45)
- **Visual:** Transition to Fairness & Governance page (`/#fairness`). Show subgroup parity tables, Blinded Model comparison, and PSI drift meters.
- **On-Screen Callout:** *"Blinded AUC: 0.6597 | Max PSI: <0.05 | p95 Latency: 386.0ms"*.
- **Narration:**
  > *"Healthcare AI demands equity. When we trained a demographic-blinded model—stripping race and gender entirely—the test AUC was 0.6597, proving our model evaluates disease severity, not demographic proxies.
  > All features exhibit Population Stability Index below 0.05, and our containerized FastAPI backend delivers sub-390 millisecond p95 latency.
  > ReadmitIQ: calibrated, explainable, and production-ready. Thank you."*
