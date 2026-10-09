# ReadmitIQ Manual Deployment Guide & Checklist

A complete, copy-paste production deployment guide for hosting the **ReadmitIQ Clinical Care-Team Command Center** (FastAPI backend + React/Vite frontend) with zero cloud resources created automatically.

---

## 1. What You Need to Prepare

Before starting, prepare the following accounts and items:

| Item | Requirement / Where to Get | Purpose | Estimated Setup Time |
|:---|:---|:---|:---:|
| **GitHub Repository** | Public or private Git repo | Source code synchronization | 2 mins |
| **Backend Host Account** | [Render.com](https://render.com) or [Hugging Face Spaces](https://huggingface.co/spaces) | FastAPI Python ML container hosting | 3 mins |
| **Frontend Host Account** | [Vercel](https://vercel.com) or [Netlify](https://netlify.com) | React SPA global CDN static hosting | 2 mins |
| **Secrets / API Keys** | **None required.** ReadmitIQ is 100% self-contained (zero LLM / external API dependencies). | N/A | 0 mins |
| **Serving Artifacts** | Artifacts bundle from `models/final/` & `reports/` | Frozen model weights, FAISS index, and demo scores | 2 mins |

**Total Estimated Deployment Time:** **12 – 18 minutes**.

---

## 2. Model Artifacts Strategy & Resource Footprints

ReadmitIQ supports two serving modes depending on host memory capacity:

| Mode | Total Disk Size | Idle RAM | Peak RAM Under Load | Description & Suitable Free Tiers |
|:---|:---:|:---:|:---:|:---|
| **LITE Mode** (`READMITIQ_SERVE_MODE=lite`) | **1.69 MB** | ~110 MB | **~195 MB** | Uses 5,000 stratified prototypes for FAISS retrieval, fast TreeSHAP, and 3,000 cohort encounters. Perfect for **Render Free Tier (512 MB)** and micro instances. |
| **FULL Mode** (`READMITIQ_SERVE_MODE=full`) | **18.47 MB** | ~190 MB | **~315 MB** | Uses full 69,535-patient FAISS index, full 14,903 cohort replay. Runs well within Render Free Tier (< 512 MB) and **Hugging Face Spaces (16 GB)**. |

### How to Supply Artifacts at Deployment Time

Choose one of the three strategies below:

### Strategy A: Git LFS (Recommended for GitHub repos)
Because the full artifact bundle is only ~18.5 MB, you can track them directly with Git LFS:
```bash
git lfs install
git lfs track "models/final/*.joblib" "models/final/*.index" "models/final/*.parquet" "reports/demo/*.parquet"
git add .gitattributes models/final/ reports/demo/
git commit -m "chore(lfs): track serving artifacts with Git LFS"
```

### Strategy B: LITE Serving Mode (Zero LFS needed, < 2 MB)
Commit the LITE artifacts directly to the repository (they are only 1.69 MB total):
```bash
# Generate / verify lite artifacts locally:
python scripts/generate_lite_artifacts.py

# Force-add the compact lite files to git:
git add -f models/final/retriever_faiss_lite.index
git add -f models/final/retriever_meta_lite.parquet
git add -f reports/demo/cohort_scored_lite.parquet
git add models/final/*.joblib models/final/*.json models/FROZEN_SELECTION.json
git commit -m "feat(artifacts): include compact lite serving artifacts"
```
Set `READMITIQ_SERVE_MODE=lite` in your environment variables.

### Strategy C: Download at Startup from Release Asset / URL
If you prefer keeping binaries strictly outside git, create a GitHub Release or upload `artifacts_bundle.tar.gz` to an S3/GCS bucket or public CDN:
Add this pre-start hook in your startup script or Dockerfile:
```bash
python -c "
import urllib.request, tarfile
url = 'https://github.com/your-org/readmitt-iq/releases/download/v1.0.0/artifacts_bundle.tar.gz'
print('Downloading ReadmitIQ artifacts...')
urllib.request.urlretrieve(url, 'artifacts.tar.gz')
with tarfile.open('artifacts.tar.gz', 'r:gz') as tar:
    tar.extractall()
print('Artifacts successfully unpacked.')
"
```

---

## 3. Backend Deployment (Choose Render or Hugging Face)

### Option 3A: Deploy to Render.com (Web Service / Docker)

1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your repository.
4. Fill in the service configuration:
   - **Name:** `readmitiq-api`
   - **Region:** Ohio (US East) or Frankfurt (EU)
   - **Branch:** `master` or `main`
   - **Runtime:** `Docker` (Render automatically uses root `Dockerfile`)
   - **Instance Type:** `Free` (512 MB RAM, 0.1 CPU)
5. Under **Advanced** $\rightarrow$ **Add Environment Variable**, configure:
   ```env
   PORT=8000
   WORKERS=1
   READMITIQ_SERVE_MODE=lite
   CORS_ORIGINS=https://readmitiq.vercel.app,http://localhost:5173
   ```
   *(Note: Set `WORKERS=1` on Free Tier to avoid RAM exhaustion; 1 worker handles ~80 req/sec).*
6. Click **Create Web Service**.
7. Wait 3–4 minutes for the Docker image to build. Once deployed, note your service URL:
   `https://readmitiq-api.onrender.com`

---

### Option 3B: Deploy to Hugging Face Spaces (Docker SDK)

1. Log into [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Fill in:
   - **Space Name:** `readmitiq-api`
   - **License:** `apache-2.0` or `mit`
   - **Space SDK:** **Docker**
   - **Docker Template:** **Blank**
   - **Space Hardware:** `CPU Basic · 2 vCPU · 16 GB RAM · Free`
4. Clone the HF Space repo or set up GitHub Actions sync:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/readmitiq-api
   git push space master:main
   ```
5. In **Settings** $\rightarrow$ **Variables and secrets**, add:
   ```env
   PORT=7860
   WORKERS=2
   READMITIQ_SERVE_MODE=full
   CORS_ORIGINS=*
   ```
6. The Space will build and serve on port 7860 at `https://<your-username>-readmitiq-api.hf.space`.

---

## 4. Frontend Deployment (Vercel)

### Deploy to Vercel (Recommended)

1. Log into [Vercel](https://vercel.com).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Select your Git repository.
4. Configure the build parameters:
   - **Framework Preset:** `Vite`
   - **Root Directory:** Click `Edit` and select `web`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm ci`
5. Expand **Environment Variables** and add:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** Your live backend URL from Step 3 without trailing slash (e.g., `https://readmitiq-api.onrender.com`).
6. Click **Deploy**.
7. In ~60 seconds, your site will be live at `https://readmitiq.vercel.app`.
8. Copy the live Vercel domain and add it to `CORS_ORIGINS` in your Render / HF dashboard if not already added.

---

## 5. Post-Deployment Verification (Copy-Paste Smoke Test)

Replace `https://your-api.onrender.com` with your live API URL:

### Test 1: Verify System Health & Mode
```bash
curl -s -f https://your-api.onrender.com/health | jq .
```
**Expected Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "serve_mode": "lite",
  "model_name": "LightGBM + Structured RAG",
  "artifacts_loaded": {
    "model": true,
    "preprocessor": true,
    "calibrator": true,
    "retriever_faiss": true,
    "retriever_meta": true,
    "feature_list": true
  }
}
```

### Test 2: Verify Calibrated Prediction & Interventions
```bash
curl -s -X POST https://your-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": "[60-70)",
    "gender": "Female",
    "time_in_hospital": 4,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 15,
    "number_inpatient": 2,
    "number_emergency": 1,
    "number_diagnoses": 8,
    "diag_1_group": "Circulatory",
    "admission_type_group": "Emergency",
    "discharge_disposition_group": "Home",
    "A1Cresult": ">8",
    "diabetesMed": "Yes",
    "insulin": "Down"
  }' | jq '{calibrated_risk: .calibrated_risk_probability, tier: .risk_tier, suggestions: .suggested_interventions.suggestions_count}'
```
**Expected Response:**
```json
{
  "calibrated_risk": 0.1471,
  "tier": "Low",
  "suggestions": 5
}
```

### Test 3: Verify Peer Precedent Retrieval
```bash
curl -s -X POST https://your-api.onrender.com/similar \
  -H "Content-Type: application/json" \
  -d '{
    "patient_features": {
      "time_in_hospital": 4,
      "number_inpatient": 2,
      "diag_1_group": "Circulatory"
    },
    "k": 20
  }' | jq '{neighbors: .k_requested, neighborhood_readmit_rate: .neighbor_readmission_rate, top_match_sim: .similar_encounters[0].similarity_score}'
```

### Test 4: Verify Retrospective Cohort Replay
```bash
curl -s "https://your-api.onrender.com/cohort?capacity=0.10" | jq '{total_cohort: .total_test_encounters, flagged: .flagged_encounters_count, lift: .lift_vs_random, precision: .precision}'
```

### Test 5: Verify Saved Model Evaluation Metrics
```bash
curl -s https://your-api.onrender.com/metrics | jq '{test_auc: .test_results.auc_roc, top10_lift: .test_results.top_10pct_capacity.lift_vs_random, brier: .test_results.brier_score_calibrated}'
```
**Expected Response:**
```json
{
  "test_auc": 0.659,
  "top10_lift": 2.09,
  "brier": 0.0973
}
```

### Test 6: Verify Frontend Integration
1. Open your Vercel URL (`https://readmitiq.vercel.app`) in your browser.
2. Verify top-right status pill displays **"API Online"** (green dot).
3. Navigate to **Cohort Triage**: adjust capacity slider from 5% to 20% and verify KPI cards update dynamically.
4. Navigate to **Patient Assessment**: click **"High Risk (Elderly)"** preset $\rightarrow$ verify risk dial animates, SHAP drivers render, and care transition suggestions appear.

---

## 6. Rollback & Troubleshooting Playbook

### Scenario A: Backend OOM (Out of Memory) Crash on Render Free Tier
- **Symptoms:** Container exits with code 137 (`Memory limit exceeded`).
- **Immediate Fix:**
  1. Open Render Dashboard $\rightarrow$ `readmitiq-api` $\rightarrow$ **Environment**.
  2. Ensure `READMITIQ_SERVE_MODE=lite` is set.
  3. Ensure `WORKERS=1` is set.
  4. Click **Save Changes** $\rightarrow$ Render will automatically restart the service using the 1.69 MB lightweight footprint (~195 MB peak RAM).

### Scenario B: CORS Errors on Frontend
- **Symptoms:** Browser console shows `Access-Control-Allow-Origin header missing`.
- **Immediate Fix:**
  1. Go to backend host settings.
  2. Set `CORS_ORIGINS` to `*` (or your exact Vercel domain e.g. `https://readmitiq.vercel.app`).
  3. Redeploy backend service.

### Scenario C: Instant 1-Click Rollback on Vercel
- If a frontend commit introduces a regression:
  1. Go to **Vercel Dashboard** $\rightarrow$ Project $\rightarrow$ **Deployments**.
  2. Locate the previous working deployment.
  3. Click `...` $\rightarrow$ **Promote to Production**. Rollback is instantaneous (< 3 seconds) with zero build delay.

### Scenario D: Instant 1-Click Rollback on Render
- If a backend image fails:
  1. Go to **Render Dashboard** $\rightarrow$ `readmitiq-api` $\rightarrow$ **Events**.
  2. Find the previous successful deploy.
  3. Click **Rollback to this deploy**.

### Scenario E: Ephemeral Container Disks on Free Hosting Tiers (Auto-Seeding on Startup)
- **Background Context:** Free hosting platforms (Render Free Tier, Railway Starter, Hugging Face Spaces free) utilize **ephemeral container disks**. When instances spin down after inactivity or restart after a code redeployment, local SQLite files (`data/readmitiq.db`) are wiped and recreated fresh.
- **Built-in Self-Healing Behavior:**
  - ReadmitIQ includes automatic startup self-seeding in `api/main.py` via `scripts/seed_demo.py:seed_if_empty()`.
  - When a container boots and detects 0 records in SQLite, it automatically runs the live scoring pipeline for 300 simulated encounters and advances the simulated Time Machine to release ~150 verified outcomes.
  - **Outcome:** Clinical evaluators, judges, and care coordinators arriving at the demo URL will **never see an empty monitoring page or empty worklist**. All seeded data is clearly labeled `source: simulated_feed`.
- **Production Persistence Option:**
  - In enterprise production environments, mount a persistent volume (e.g. Render Disk or AWS EBS) and configure the environment variable:
    ```bash
    READMITIQ_DB_PATH=/var/data/readmitiq.db
    READMITIQ_AUTO_SEED=false
    ```

