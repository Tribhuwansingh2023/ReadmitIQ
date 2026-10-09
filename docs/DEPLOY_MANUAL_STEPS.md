# ReadmitIQ Manual Deployment Guide

This document outlines the exact, step-by-step procedure for deploying the **ReadmitIQ API Backend** and **Web Command Center** to cloud hosting using your own accounts and credentials.

---

## 1. Artifacts & Resource Footprint

The frozen model artifacts are git-ignored to keep the repository lightweight. Before deploying the backend, ensure these artifacts are present in the runtime environment:

| File Path | Description | Size | Memory Impact |
|:---|:---|:---:|:---:|
| `models/final/model.joblib` | LightGBM final classifier | ~282 KB | Low (<10 MB) |
| `models/final/calibrator.joblib` | Isotonic regression calibrator | ~2.3 KB | Negligible |
| `models/final/preprocessor_base.joblib` | 41-feature tabular pipeline | ~11 KB | Negligible |
| `models/final/preprocessor.joblib` | 42-feature pipeline with RAG | ~12 KB | Negligible |
| `models/final/structured_svd.joblib` | 64-dim TruncatedSVD projection | ~38 KB | Negligible |
| `models/final/retriever_faiss.index` | 64-dim FAISS Inner Product index (69k vectors) | ~17.8 MB | Low (~25 MB in RAM) |
| `models/final/retriever_meta.parquet` | Anonymized train metadata for neighbor retrieval | ~945 KB | Low (~8 MB in RAM) |
| `reports/demo/cohort_scored.parquet` | Precomputed test scores for retrospective replay | ~850 KB | Low (~6 MB in RAM) |

**Total Artifact Footprint:** ~20 MB on disk; **Total Runtime RAM:** ~220 MB idle, ~350 MB under concurrent load.

---

## 2. Backend Deployment (FastAPI)

### Option A: Render.com (Recommended)
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Blueprint** or **Web Service**.
3. Connect your GitHub repository.
4. If using `render.yaml`, Render automatically parses the configuration:
   - **Environment:** Python 3.11
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. **Artifacts Upload:**
   - Because `models/final/` is in `.gitignore`, use the Render persistent disk OR upload the 20MB `models/final/` directory into a private storage bucket (AWS S3, Google Cloud Storage, or Hugging Face Models repo) and add a pre-download step in your build command:
     ```bash
     python -c "import urllib.request; ..." # or use curl/wget
     ```
6. **Environment Variables:**
   - `CORS_ORIGINS`: `https://your-frontend.vercel.app`
   - `PORT`: `8000` (assigned automatically by Render)
7. **Free-Tier RAM Note:** Render Free Tier offers 512 MB RAM. The FastAPI service with FAISS runs comfortably within ~350 MB. If spikes occur, upgrade to the Starter Plan ($7/mo with 2 GB RAM).

### Option B: Hugging Face Spaces (Docker)
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Select **Docker** SDK and **Blank** template.
3. Push the repository including `Dockerfile`.
4. Upload `models/final/` to the Space (free tier provides 16 GB RAM and 2 vCPUs).
5. The API will be accessible at `https://<username>-readmitiq.hf.space`.

---

## 3. Frontend Deployment (React + Vite SPA)

### Deploying to Vercel
1. Log into [Vercel](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your Git repository.
4. Configure Project Settings:
   - **Framework Preset:** Vite
   - **Root Directory:** `web`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
5. **Environment Variables:**
   - Set `VITE_API_BASE_URL` to your live backend URL (e.g. `https://readmitiq-api.onrender.com` without trailing slash).
6. Click **Deploy**. Vercel will build and provide a global CDN URL within 60 seconds.

---

## 4. Verification & Smoke Test

Once deployed, run these commands to verify the production setup:

```bash
# 1. Verify API Health
curl https://your-api-url.onrender.com/health

# Expected response:
# {"status":"healthy","version":"1.0.0","model_name":"LightGBM + Structured RAG",...}

# 2. Verify Cohort Replay
curl "https://your-api-url.onrender.com/cohort?capacity=0.10"

# 3. Verify Frontend
# Open your Vercel URL in a browser. Ensure the top-right status badge displays "API Online" (green dot).
```
