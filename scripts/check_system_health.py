"""Ping live ReadmitIQ FastAPI backend and verify health endpoint, model version headers, and response latency.

Usage:
    python scripts/check_system_health.py [base_url]
"""

import json
import sys
import time
import urllib.request

DEFAULT_URL = "http://localhost:8000"


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    health_url = f"{base_url.rstrip('/')}/health"

    print(f"Pinging ReadmitIQ Gateway at: {health_url}")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            version_hdr = resp.headers.get("X-Model-Version", "Missing")

            print(f"[PASS] Status Code: {resp.status} (in {elapsed_ms:.2f} ms)")
            print(f"[PASS] System Status: {data.get('status')}")
            print(f"[PASS] Serve Mode: {data.get('serve_mode')}")
            print(f"[PASS] X-Model-Version: {version_hdr}")
            print(f"[PASS] Model Name: {data.get('model_name')}")
            print("\nLoaded Runtime Artifacts:")
            for k, v in data.get("artifacts_loaded", {}).items():
                print(f"  - {k}: {'LOADED' if v else 'NOT LOADED'}")
    except Exception as e:
        print(f"[ERROR] Could not connect to {health_url}: {e}")
        print("Ensure the FastAPI backend is running via `scripts/run_backend.bat` or `python -m uvicorn api.main:app`.")
        sys.exit(1)


if __name__ == "__main__":
    main()
