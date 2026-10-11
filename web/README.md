# ReadmitIQ Web Frontend

Production clinical decision support user interface for the **ReadmitIQ ActionLoop** platform.

- **Author & Maintainer:** Tribhuwan Singh ([@Tribhuwansingh2023](https://github.com/Tribhuwansingh2023), `webosingh93@gmail.com`)
- **Technology Stack:** React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons, JetBrains Mono & Fraunces typography.
- **Key Clinical Interfaces:**
  - **Capacity-Aware Cohort Triage Desk:** Dynamic outreach quota thresholding with deterministic raw-score tie-breaking.
  - **Clinical Explainability Engine (XAI):** Log-odds TreeSHAP attribution, non-causal plain-English clinical summaries, 5-point reliability matrix, and global validation cohort reference baselines.
  - **What-If Sensitivity Explorer:** Counterfactual clinical simulation with live TreeSHAP attribution shift ($\Delta \phi_i$) calculation.
  - **Precedent Orbit Evidence:** Similar patient inspection via 64-dim FAISS inner-product latent projections.
  - **Trust & Safety Surveillance Monitor:** Real-time PSI drift monitoring and data quality audits.

## Development Setup

```bash
npm install
npm run dev
```

## Production Build & Typecheck

```bash
npm run build
```
