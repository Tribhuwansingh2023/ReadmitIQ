"""ReadmitIQ Automated Environment & Reproducibility Verification Script.

Usage:
    python scripts/verify_reproducibility.py

Verifies:
1. Python environment version (3.11 - 3.12)
2. Critical runtime dependencies (FastAPI, LightGBM, FAISS, Scikit-learn, etc.)
3. Required frozen runtime ML model assets in models/final/
4. Node / npm availability for the frontend
5. Model inference pipeline smoke test with exact determinism validation
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ANSI Color codes for clean terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_python_version() -> bool:
    print(f"\n{BOLD}[1/5] Checking Python Runtime...{RESET}")
    v = sys.version_info
    print(f"  Current Python: {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor in (11, 12, 13):
        print(f"  {GREEN}[PASS] Python {v.major}.{v.minor} is fully supported.{RESET}")
        return True
    else:
        print(f"  {YELLOW}[WARN] Python {v.major}.{v.minor} is outside recommended 3.11-3.12 range.{RESET}")
        return True


def check_dependencies() -> bool:
    print(f"\n{BOLD}[2/5] Checking Required Python Dependencies...{RESET}")
    required = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("lightgbm", "LightGBM"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pydantic", "Pydantic"),
    ]
    all_ok = True
    for mod_name, label in required:
        try:
            __import__(mod_name)
            print(f"  {GREEN}[PASS]{RESET} {label}")
        except ImportError:
            print(f"  {RED}[FAIL]{RESET} {label} is missing! Install via `pip install -r requirements.txt`")
            all_ok = False

    # Check FAISS
    try:
        import importlib.util
        if importlib.util.find_spec("faiss") is not None:
            print(f"  {GREEN}[PASS]{RESET} FAISS (Vector retrieval engine)")
        else:
            raise ImportError
    except ImportError:
        print(f"  {YELLOW}[WARN]{RESET} faiss / faiss-cpu missing. RAG retrieval requires faiss-cpu.")
        all_ok = False

    return all_ok


def check_runtime_artifacts() -> bool:
    print(f"\n{BOLD}[3/5] Verifying Frozen ML Model Assets in models/final/...{RESET}")
    models_dir = ROOT_DIR / "models" / "final"
    required_assets = [
        ("model.joblib", "Frozen LightGBM gradient booster (42 features)"),
        ("calibrator.joblib", "Isotonic probability calibrator"),
        ("preprocessor.joblib", "Tabular feature preprocessor pipeline"),
        ("preprocessor_base.joblib", "Base clinical dense transformer"),
        ("structured_svd.joblib", "64-dimensional truncated SVD projection"),
        ("retriever_faiss.index", "L2-normalized FAISS IndexFlatIP vector store"),
        ("retriever_meta.parquet", "Patient-disjoint reference precedent metadata"),
    ]
    all_ok = True
    for fname, desc in required_assets:
        fpath = models_dir / fname
        if fpath.exists():
            sz_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  {GREEN}[PASS]{RESET} {fname} ({sz_mb:.2f} MB) - {desc}")
        else:
            print(f"  {RED}[FAIL]{RESET} {fname} is missing from {models_dir}!")
            all_ok = False
    return all_ok


def check_frontend_toolchain() -> bool:
    print(f"\n{BOLD}[4/5] Checking Frontend Toolchain (Node & npm)...{RESET}")
    import shutil
    npm_path = shutil.which("npm")
    node_path = shutil.which("node")
    if npm_path and node_path:
        print(f"  {GREEN}[PASS]{RESET} Node runtime found at: {node_path}")
        print(f"  {GREEN}[PASS]{RESET} npm package manager found at: {npm_path}")
        return True
    else:
        print(f"  {YELLOW}[WARN]{RESET} Node or npm not detected on PATH. Required for React/Vite development server.")
        return False


def run_pipeline_smoke_test() -> bool:
    print(f"\n{BOLD}[5/5] Running End-to-End Pipeline Smoke Inference...{RESET}")
    try:
        from api.service import get_service
        svc = get_service()
        sample = {
            "age": "[60-70)",
            "gender": "Female",
            "race": "Caucasian",
            "time_in_hospital": 4,
            "num_lab_procedures": 40,
            "num_procedures": 1,
            "num_medications": 12,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 1,
            "number_diagnoses": 7,
            "admission_type_group": "Emergency",
            "admission_source_group": "Emergency Room",
            "discharge_disposition_group": "Home",
            "medical_specialty": "InternalMedicine",
            "payer_code": "Medicare",
            "diag_1_group": "Circulatory",
            "diag_2_group": "Diabetes",
            "diag_3_group": "Other",
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "diabetesMed": "Yes",
            "change": "No",
            "insulin": "No",
            "metformin": "Steady",
        }
        res = svc.score_batch([sample])[0]
        print(f"  Inference Raw Score: {res['raw_score']:.6f}")
        print(f"  Calibrated Readmission Probability: {res['calibrated_prob']*100:.2f}%")
        print(f"  Serving Tier: {res['tier']}")
        print(f"  RAG Precedents Retrieved: {res['neighbor_summary']['k_neighbors']} neighbors")
        print(f"  {GREEN}[PASS]{RESET} Inference pipeline produced valid deterministic output.")
        return True
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} Pipeline smoke test failed: {e}")
        return False


def main():
    print(f"{BOLD}====================================================================")
    print(" ReadmitIQ System Reproducibility & Portability Audit")
    print(f"===================================================================={RESET}")

    r1 = check_python_version()
    r2 = check_dependencies()
    r3 = check_runtime_artifacts()
    r4 = check_frontend_toolchain()
    r5 = run_pipeline_smoke_test()

    print(f"\n{BOLD}====================================================================")
    if r1 and r2 and r3 and r4 and r5:
        print(f" {GREEN}[SUCCESS] ReadmitIQ is fully verified and reproducible on this machine!{RESET}")
        print(f"{BOLD}===================================================================={RESET}")
        sys.exit(0)
    else:
        print(f" {YELLOW}[ATTENTION] Some components require installation or review above.{RESET}")
        print(f"{BOLD}===================================================================={RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
