# ReadmitIQ Frequently Asked Questions (FAQ)

Answers to common architectural, clinical, and data science questions about ReadmitIQ.

---

### Q1: Why does ReadmitIQ use 42 features instead of hundreds of raw EHR variables?
**A:** Clinical utility and feature reliability. The 42 features represent verified clinical categories: length of stay, prior utilization (inpatient, emergency, outpatient), diagnosis groupings (CCS collapsed), glycemic laboratory markers (HbA1c, peak glucose), and active medications. Restricting to these 42 features prevents extreme sparsity, reduces overfitting on institutional idiosyncrasies, and enables fast sub-millisecond inference and explainability on commodity hardware.

---

### Q2: Why Isotonic Regression instead of Platt (Logistic) Scaling?
**A:** Platt scaling assumes that the log-odds of the raw model score map monotonically and linearly to the true posterior probability. Gradient-boosted decision trees often exhibit piecewise-constant or non-linear calibration plateaus. Isotonic regression is non-parametric and fits a flexible isotonic step function, reducing test set Brier score by 59.2% without assuming a parametric sigmoid shape.

---

### Q3: Why rank worklists by raw model score instead of calibrated probability?
**A:** Isotonic calibration maps intervals of raw scores to flat horizontal steps (plateaus). When multiple patients fall on the same calibration step, they receive identical calibrated probabilities (e.g. $14.71\%$). If a worklist sorted by calibrated probability, dozens of patients would tie arbitrarily. ReadmitIQ sorts by `raw_score DESC` to preserve continuous ordering, then displays the calibrated probability to the clinician for transparent risk communication.

---

### Q4: Does RAG neighbor retrieval increase model predictive accuracy?
**A:** No, and ReadmitIQ is completely transparent about this. In paired bootstrap cross-validation on the held-out split, adding structural neighbor rates yielded $\Delta\text{AUC} = 0.0000$ (with confidence intervals crossing zero). Precedent retrieval is included for **clinical explainability and case context**, enabling nurse coordinators to see how historically similar patients were treated and whether they were readmitted.

---

### Q5: How does the Time Machine work in the simulated feed?
**A:** When the simulated replay feed is running, historical inpatient encounters from the held-out partition are ingested in real time. Their true 30-day readmission outcomes remain sealed in the database until the user advances the Time Machine by $+7\text{d}$ or $+30\text{d}$. Once outcomes are unsealed, the Trust & Safety Monitor evaluates live empirical discrimination (ROC-AUC, Brier score) against ground truth.

---

### Q6: Can clinicians override machine predictions?
**A:** Yes. ReadmitIQ enforces human-in-the-loop decision-making. Clinicians can upgrade or downgrade an encounter's risk tier directly in the Patient Decision Brief. All overrides mandate entering a clinical rationale and are recorded in the immutable audit log.
