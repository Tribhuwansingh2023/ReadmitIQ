# ReadmitIQ Evaluation & Scientific Benchmark Results

All evaluation metrics reported in ReadmitIQ are derived from strictly partitioned, patient-disjoint held-out data splits. Models, parameters, and thresholds are completely frozen.

---

## 1. Dataset Partitioning Strategy
- **Source Cohort:** Diabetic Inpatient Encounter Dataset ($N = 101,766$ records).
- **Patient Disjoint Splitting:** Partitioned by patient identifier (`patient_nbr`) ensuring zero encounters from the same patient appear across splits:
  - **Train Split:** 69,535 encounters (68.3%)
  - **Validation Split:** 17,328 encounters (17.0%)
  - **Test Benchmark Split:** 14,903 encounters (14.7%)
- **Data Leakage Audits:** 4 independent automated leakage audits (`test_audit_leakage.py`, `test_rag_leakage.py`) enforce zero overlap in identifiers, embeddings, or post-discharge outcomes.

---

## 2. Benchmark Validation Performance
Evaluated on the frozen held-out validation cohort ($n = 17,328$):

| Model Architecture | Features | ROC-AUC [95% CI] | Brier Score | Paired $\Delta\text{AUC}$ vs Primary |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM Primary (Frozen)** | 42 | **0.6731 [0.6602, 0.6863]** | **0.0894** | Benchmark Baseline |
| LightGBM + RAG Augmented | 43 | 0.6731 [0.6602, 0.6863] | 0.0894 | 0.0000 (CI crosses zero) |
| Blinded LightGBM (No Race/Gender) | 40 | 0.6738 [0.6609, 0.6870] | 0.0893 | +0.0007 |
| Logistic Regression (Baseline) | 42 | 0.6552 [0.6420, 0.6684] | 0.0912 | -0.0179 |

---

## 3. Capacity Thresholds & Operating Points
Derived from the empirical cumulative distribution function (CDF) of model predictions on the reference cohort:

| Capacity Allocation Tier | Population Share | Raw Score Cutoff | Calibrated Probability Cutoff |
| :--- | :--- | :--- | :--- |
| **Critical Tier** | Top 5% of risk | $\ge 0.6839$ | $\ge 19.4\%$ |
| **High Tier** | Top 10% of risk | $\ge 0.6408$ | $\ge 16.2\%$ |
| **Medium Tier** | Top 20% of risk | $\ge 0.5793$ | $\ge 12.8\%$ |
| **Low Tier** | Remaining 80% | $< 0.5793$ | $< 12.8\%$ |

- **Optimal F1 Decision Threshold:** $0.1245$ (chosen to balance recall and precision for proactive clinical nurse call outreach).
