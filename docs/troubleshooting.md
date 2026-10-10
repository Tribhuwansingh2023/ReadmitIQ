# ReadmitIQ Troubleshooting Guide

Common issues encountered when setting up ReadmitIQ on a new machine and how to resolve them.

---

## 1. FAISS / Vector Index Issues

### Problem: `ModuleNotFoundError: No module named 'faiss'`
- **Cause:** FAISS binary wheels vary across operating systems.
- **Solution:**
  - On **Windows / macOS (Apple Silicon / Intel) / Linux**:
    ```bash
    pip install faiss-cpu
    ```
  - If using Conda:
    ```bash
    conda install -c conda-forge faiss-cpu
    ```

### Problem: `RuntimeError: Error in virtual void faiss::Index::search...`
- **Cause:** Dimension mismatch in vector query.
- **Solution:** Ensure `models/final/structured_svd.joblib` and `models/final/retriever_faiss.index` are both tracked. Run `python scripts/verify_reproducibility.py` to check asset sizes.

---

## 2. Port Conflicts

### Problem: `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)` (Windows)
- **Cause:** Another process is occupying port 8000.
- **Solution:**
  - Find process on port 8000:
    ```powershell
    netstat -ano | findstr :8000
    ```
  - Kill process by PID or run ReadmitIQ on an alternative port:
    ```bash
    python -m uvicorn api.main:app --port 8080 --reload
    ```
  - Update `web/.env` or `VITE_API_BASE_URL` to `http://localhost:8080`.

### Problem: `Port 5173 is in use, trying another one...`
- **Solution:** Vite automatically falls back to 5174. If you want to force 5173, terminate other running Vite instances.

---

## 3. Scikit-learn Version Warnings

### Problem: `InconsistentVersionWarning: Trying to unpickle estimator...`
- **Cause:** Model was pickled under scikit-learn 1.9.0/1.9.1.
- **Impact:** Informational warning only; numerical outputs and predictions are unaffected.
- **Solution:** Standard behavior during deserialization; tests and smoke inference verify exact numerical determinism.

---

## 4. Frontend Compilation & Vite Build

### Problem: `Cannot find name 'X'` or TypeScript compilation errors
- **Solution:**
  ```bash
  cd web
  rm -rf node_modules package-lock.json
  npm install
  npm run build
  ```
  Ensure Node version is $\ge 18$.
