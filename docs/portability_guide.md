# ReadmitIQ Portability & Setup Runbook

This runbook guides another developer or clinician through cloning and running ReadmitIQ on a fresh laptop (Windows, macOS, or Linux).

---

## 1. Prerequisites
- **Python:** 3.11 or 3.12 (Python 3.12 recommended).
- **Node.js:** v18, v20, or v22 with `npm`.
- **Git:** Standard git client.
- **Hardware:** $\ge 4 \text{ GB RAM}$ (runs smoothly on lightweight laptops).

---

## 2. Clone the Repository
```bash
git clone https://github.com/Tribhuwansingh2023/ReadmitIQ.git
cd ReadmitIQ
```

---

## 3. Python Backend Setup

### Option A: Using `venv` (Standard Python)
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
# OR on Windows (Command Prompt)
.venv\Scripts\activate.bat
# OR on macOS / Linux (POSIX)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Using `uv` (Fast Package Manager)
```bash
uv venv .venv
# Activate virtual environment per your OS, then:
uv pip install -r requirements.txt
```

---

## 4. Node.js Frontend Setup
```bash
cd web
npm install
cd ..
```

---

## 5. Verify Installation
Run the automated reproducibility verification tool:
```bash
python scripts/verify_reproducibility.py
```
Expected output:
```
[1/5] Checking Python Runtime...           [PASS]
[2/5] Checking Required Python Dependencies... [PASS]
[3/5] Verifying Frozen ML Model Assets...   [PASS]
[4/5] Checking Frontend Toolchain...       [PASS]
[5/5] Running Smoke Inference...           [PASS]
[SUCCESS] ReadmitIQ is fully verified and reproducible on this machine!
```

---

## 6. Running the Application

### Starting the Backend (FastAPI)
Using launcher script:
- **Windows:** `scripts\run_backend.bat`
- **Linux/macOS:** `chmod +x scripts/run_backend.sh && ./scripts/run_backend.sh`

Or directly via CLI:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Starting the Frontend (React / Vite)
Using launcher script:
- **Windows:** `scripts\run_frontend.bat`
- **Linux/macOS:** `chmod +x scripts/run_frontend.sh && ./scripts/run_frontend.sh`

Or directly via CLI:
```bash
cd web
npm run dev
```
Open your browser to: [http://localhost:5173](http://localhost:5173)

---

## 7. Running with Docker Compose (Containerized)
For fully containerized deployment:
```bash
docker-compose up --build
```
- **Web UI:** [http://localhost:5173](http://localhost:5173)
- **Backend API:** [http://localhost:8000](http://localhost:8000)

---

## 8. Running Automated Test Suites
```bash
# Run all 80 tests
pytest

# Run ActionLoop integration tests specifically
pytest tests/test_actionloop.py -v

# Run frontend type-check & production build
cd web && npm run build
```
