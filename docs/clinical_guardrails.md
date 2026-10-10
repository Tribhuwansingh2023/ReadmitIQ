# ReadmitIQ Clinical Guardrails & Methodological Disclosure

## Research Prototype Disclaimer
ReadmitIQ is developed as an investigational decision support prototype for 30-day all-cause hospital readmission risk stratification and evidence-supported care coordination. It is **not** a cleared Medical Device (SaMD) under FDA 510(k), De Novo, or EU MDR regulations. It does **not** provide independent diagnostic decisions or autonomous treatment directives.

---

## Methodological Guardrails

### 1. Precedent Retrieval Does Not Add Independent Predictive Lift
- **Observation:** Retrieval of similar historical patients provides valuable case context and explainability to human care coordinators.
- **Scientific Evidence:** Rigorous paired cross-validation and held-out test evaluation establish that augmenting tabular gradient boosted models with structural neighbor rates yields:
  $$\Delta\text{ROC-AUC} = 0.0000 \quad [95\% \text{ CI crosses zero}]$$
- **Policy:** ReadmitIQ openly discloses this empirical finding in the UI and documentation. We reject fabricated claims of predictive lift that the scientific evidence does not support.

### 2. Sensitivity Analysis vs. Causal Interventions
- **Observation:** The What-If Risk Explorer allows users to simulate how risk predictions shift under modified feature inputs.
- **Scientific Boundary:** Gradient-boosted decision trees capture observational correlations across EHR data. They do **not** estimate causal treatment effects ($ATE$). Modifying length of stay from 8 days to 3 days in the explorer shows how the algorithm evaluates that patient profile; it does not guarantee that early discharge reduces true readmission risk.
- **Policy:** The interface permanently attaches an explicit sensitivity disclaimer and does not label simulations as treatment recommendations.

### 3. Small-n Statistical Power Guardrails
- **Observation:** Hospital systems often track live discrimination or subgroup parity on small prospective sample sizes.
- **Statistical Vulnerability:** Point estimates for ROC-AUC, Brier score, or False Positive Rates computed on $n < 50$ or $n < 100$ patients possess confidence intervals spanning almost the entire metric range (e.g., $[0.42, 0.94]$).
- **Policy:**
  - Live empirical ROC-AUC is withheld until sample sizes reach $n \ge 200$ resolved outcomes with $\ge 20$ positive readmissions.
  - Subgroup audits with $n < 100$ encounters are visibly marked as **Unpowered Small-n Guardrails** to prevent premature or misleading compliance benchmarks.

### 4. Human-in-the-Loop Clinical Override
- Machine predictions serve solely as triage recommendations.
- Frontline coordinators and attending clinicians maintain the absolute authority to adjust risk tiers or override priority queues.
- Every clinical override mandates a documented clinical rationale preserved in immutable audit logs.
