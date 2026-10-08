# ReadmitIQ Overnight Autonomous Execution Log

Running log of automated remediation, model freeze, evaluation, API, and web development.

| Timestamp (UTC) | Stage / Block | Status | Key Metrics / Findings | Notes & Decisions Made |
|:---|:---|:---:|:---|:---|
| 2026-10-08 21:52 | Block 1: Preprocessing & Features | DONE | Column count: 141 (target <150). Base LR Val AUC: 0.6611. 11/11 pytest passed. | Collapsed medical_specialty (top 10 + Missing + Other) and payer_code (<1% to Other). Aligned config split n_splits=10. Saved preprocessor_base.joblib. |
| 2026-10-08 22:04 | Block 2: Notes & Full Retrieval | DONE | Train RAG: 69,535 rows, Val RAG: 14,902 rows, Test RAG: 14,903 rows. Struct RAG k=20 Val Corr: 0.1451, Val AUC: 0.6039 (k=50: 0.6207). 12/12 pytest passed. | Switched notes to 1 uniform template (0 leakage across all 99k notes). Built 64-dim Structured FAISS retriever with 5-fold patient-grouped OOF. Vectorized retrieval with 0 iloc loops. Prepared Colab GPU embedding notebook. Local CPU note embedding paused to prevent 8GB RAM thrashing. |
