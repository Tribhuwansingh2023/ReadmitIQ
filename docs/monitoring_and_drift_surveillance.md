# ReadmitIQ Operational Monitoring & Drift Surveillance

ReadmitIQ implements real-time operational surveillance to detect statistical covariate shifts and model degradation before patient care is compromised.

---

## 1. Population Stability Index (PSI)

### Mathematical Formulation
The Population Stability Index measures the degree of divergence between an observed empirical probability distribution $A$ (actual live serving stream) and a reference distribution $E$ (expected training baseline):

$$\text{PSI} = \sum_{b=1}^{B} \left( A_b - E_b \right) \times \ln\left(\frac{A_b}{E_b}\right)$$

Where:
- $B$: Number of reference bins (typically 10 deciles).
- $A_b$: Proportion of live observations falling in bin $b$.
- $E_b$: Expected proportion of reference observations falling in bin $b$ (e.g., $10\%$ per decile).

### Interpretation Thresholds
- $\text{PSI} < 0.10$: **Stable (Minimal Shift)** — No population changes requiring retraining.
- $0.10 \le \text{PSI} < 0.25$: **Moderate Shift** — Subpopulation dynamics are altering; heightened clinical monitoring advised.
- $\text{PSI} \ge 0.25$: **High Drift Alert** — Significant divergence from training baseline; automated alerts trigger in the dashboard.

---

## 2. Decile Reference Edges
The model's serving thresholds file (`reports/metrics/serving_thresholds.json`) records the exact 10 decile bin edges computed over the 69,535 training encounters:
- Bins partition raw model prediction scores.
- Incoming live scores are binned dynamically into these fixed reference bins without altering the underlying models.

---

## 3. The Time Machine (Outcome Simulation Feed)
- **Problem:** Evaluating live model performance (such as true ROC-AUC or Brier score) requires observing patient readmission status after 30 calendar days have elapsed.
- **Solution:** ReadmitIQ features an interactive **Time Machine** endpoint (`POST /v1/feed/advance-time`). Advancing the clock by $+7\text{d}$ or $+30\text{d}$ unseals ground-truth clinical outcomes from historical test partitions.
- **Empirical Tracking:** Once $> 200$ outcomes are unsealed, live empirical ROC-AUC, 95% bootstrap confidence intervals, and calibration curves are rendered in real time.
