# ReadmitIQ Fairness & Demographic Subgroup Audit

A comprehensive fairness and subgroup parity audit was conducted on the held-out test partition ($n = 14,903$) across demographic slices, evaluated at the top 10% capacity triage cutoff ($0.194$).

---

## 1. Race & Ethnicity Audit

| Subgroup | Sample Size ($n$) | Prevalence | ROC-AUC [95% CI] | Selection Rate | True Positive Rate (TPR) | Statistical Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Caucasian (Ref)** | 11,188 | 11.4% | 0.657 [0.643, 0.674] | 12.7% | 24.7% | **Adequately Powered** |
| **African American** | 2,782 | 12.2% | 0.648 [0.617, 0.677] | 15.3% | 29.6% | **Adequately Powered** |
| **Hispanic** | 302 | 9.6% | 0.665 [0.551, 0.772] | 10.9% | 27.6% | **Adequately Powered** |
| **Asian** | 79 | 6.3% | 0.710 [0.418, 0.939] | 6.3% | 20.0% | **Unpowered Small-n Guard** ($n < 100$) |
| **Other / Unknown** | 552 | 8.9% | 0.688 [0.612, 0.764] | 8.5% | 20.4% | **Adequately Powered** |

---

## 2. Age Bracket Audit

| Age Group | Sample Size ($n$) | Prevalence | ROC-AUC [95% CI] | Selection Rate | TPR | Statistical Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[70-80) (Ref)** | 3,745 | 12.4% | 0.649 [0.622, 0.676] | 13.9% | 26.2% | **Adequately Powered** |
| **[60-70)** | 3,365 | 11.8% | 0.651 [0.623, 0.679] | 13.2% | 25.4% | **Adequately Powered** |
| **[50-60)** | 2,492 | 10.7% | 0.663 [0.629, 0.697] | 12.4% | 26.3% | **Adequately Powered** |
| **[80-90)** | 2,491 | 11.2% | 0.644 [0.609, 0.679] | 13.6% | 25.1% | **Adequately Powered** |
| **[40-50)** | 1,412 | 10.1% | 0.678 [0.631, 0.725] | 11.2% | 28.7% | **Adequately Powered** |
| **[30-40)** | 549 | 9.3% | 0.684 [0.598, 0.770] | 9.8% | 27.5% | **Adequately Powered** |
| **[20-30)** | 239 | 10.9% | 0.671 [0.548, 0.794] | 11.7% | 30.8% | **Adequately Powered** |
| **[10-20)** | 98 | 6.1% | 0.685 [0.442, 0.928] | 7.1% | 16.7% | **Unpowered Small-n Guard** ($n < 100$) |
| **[0-10)** | 22 | 0.0% | N/A | 0.0% | 0.0% | **Unpowered Small-n Guard** ($n < 100$) |

---

## 3. Small-n Guardrail Disclosure
In both the Asian racial category ($n = 79$) and adolescent age brackets ($n = 98$ and $n = 22$), sample size falls below the power threshold of $100$ encounters. Confidence intervals on discrimination span over 50 percentage points. ReadmitIQ explicitly tags these slices with an **Unpowered Small-n Guard**, preventing unsafe comparative claims.
