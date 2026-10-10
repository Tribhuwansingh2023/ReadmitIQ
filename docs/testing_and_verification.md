# ReadmitIQ Test Suite & Verification Guide

ReadmitIQ includes an extensive automated test suite covering unit tests, API integration, scientific leakage audits, explainability correctness, and ActionLoop workflow lifecycle.

---

## 1. Test Suite Structure

The test suite comprises **80 automated tests** located in `tests/`:

| Test Module | Tests | Scope & Purpose |
| :--- | :--- | :--- |
| `tests/test_actionloop.py` | 8 | End-to-end ActionLoop Stages 1-5, What-If explorer, and Trust & Safety monitor |
| `tests/test_v1_workflow.py` | 11 | Desk worklist, sorting, task persistence, overrides, and CSV ingestion |
| `tests/test_api.py` | 7 | FastAPI endpoints, health checks, version headers, and input schemas |
| `tests/test_audit_leakage.py` | 4 | Patient-disjoint split verification and zero cross-split patient overlap |
| `tests/test_rag_leakage.py` | 5 | RAG vector index split isolation (ensures test encounters are never indexed) |
| `tests/test_explain.py` | 13 | TreeSHAP exactness, additivity sum property, and top-driver generation |
| `tests/test_scoring_correctness.py` | 3 | Deterministic inference repeatability across batch vs single scoring |
| `tests/test_features.py` | 5 | Feature transformations, CCS diagnostic grouping, and categorical encoding |
| `tests/test_interventions.py` | 6 | Rule-based clinical care suggestion generation from top SHAP risk factors |
| `tests/test_stage2_phase4.py` | 5 | Legacy integration compatibility and regression protection |
| `tests/test_transformer_import.py` | 10 | Scikit-learn transformer compatibility and pipeline unpickling integrity |
| `tests/test_provenance_paths.py` | 1 | Metadata provenance references and documentation traceability |
| `tests/test_data_split.py` | 2 | Dataset row counts and target label distribution preservation |

---

## 2. Running Test Commands

### Full Test Suite
```bash
pytest
```
*Expected: 80 passed in ~80s.*

### ActionLoop Integration Tests
```bash
pytest tests/test_actionloop.py -v
```

### Leakage and Split Isolation Audits
```bash
pytest tests/test_audit_leakage.py tests/test_rag_leakage.py -v
```

---

## 3. Automated Reproducibility Verifier
Run the standalone portability verification script:
```bash
python scripts/verify_reproducibility.py
```
This checks:
1. Python version (3.11 - 3.12).
2. Required packages (FastAPI, LightGBM, FAISS, Scikit-learn, Pandas, NumPy, Pydantic).
3. Frozen model files in `models/final/` (all 7 files present and verified).
4. Node & npm presence for the frontend.
5. Live deterministic pipeline smoke inference test.
