"""Benchmark latency of API /predict endpoint locally.

Measures p50, p90, p95, p99 latencies across 100 realistic requests
including FAISS retrieval, LightGBM prediction, calibration, SHAP attribution,
and deterministic decision support generation.
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from api.main import app


def run_latency_benchmark(n_requests: int = 100) -> dict:
    print(f"Starting API /predict latency benchmark ({n_requests} sequential requests)...")
    client = TestClient(app)

    # Health check warmup
    warmup_res = client.get("/health")
    assert warmup_res.status_code == 200, "Health check failed during benchmark warmup"

    # Diverse test payloads representing realistic clinical variance
    base_payloads = [
        {
            "age": "[70-80)",
            "gender": "Female",
            "time_in_hospital": 7,
            "number_inpatient": 3,
            "number_emergency": 2,
            "num_medications": 24,
            "diag_1_group": "Circulatory",
            "A1Cresult": ">8",
            "insulin": "Down",
            "change": "Ch",
        },
        {
            "age": "[60-70)",
            "gender": "Male",
            "time_in_hospital": 4,
            "number_inpatient": 1,
            "number_emergency": 0,
            "num_medications": 14,
            "diag_1_group": "Diabetes",
            "A1Cresult": "None",
            "insulin": "Steady",
            "change": "No",
        },
        {
            "age": "[40-50)",
            "gender": "Female",
            "time_in_hospital": 2,
            "number_inpatient": 0,
            "number_emergency": 0,
            "num_medications": 7,
            "diag_1_group": "Other",
            "A1Cresult": "Norm",
            "insulin": "No",
            "metformin": "Steady",
            "change": "No",
        },
        {
            "age": "[60-70)",
            "gender": "Male",
            "time_in_hospital": 5,
            "number_inpatient": 1,
            "number_emergency": 0,
            "num_medications": 18,
            "diag_1_group": "Respiratory",
            "A1Cresult": "None",
            "insulin": "No",
            "change": "No",
        },
    ]

    latencies_ms = []

    # Warmup request
    client.post("/predict", json=base_payloads[0])

    for i in range(n_requests):
        payload = base_payloads[i % len(base_payloads)]
        t0 = time.perf_counter()
        res = client.post("/predict", json=payload)
        t1 = time.perf_counter()
        assert res.status_code == 200, f"Request {i} failed: {res.text}"
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_sorted = sorted(latencies_ms)
    p50 = statistics.median(latencies_sorted)
    p90 = latencies_sorted[int(n_requests * 0.90)]
    p95 = latencies_sorted[int(n_requests * 0.95)]
    p99 = latencies_sorted[int(n_requests * 0.99)]
    mean_lat = statistics.mean(latencies_sorted)
    max_lat = max(latencies_sorted)

    results = {
        "n_requests": n_requests,
        "mean_ms": round(mean_lat, 2),
        "median_p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(max_lat, 2),
        "under_500ms_target": bool(p95 < 500.0),
    }

    print("\n=== LATENCY BENCHMARK RESULTS ===")
    print(f"Mean Latency:        {results['mean_ms']} ms")
    print(f"p50 (Median):        {results['median_p50_ms']} ms")
    print(f"p90:                 {results['p90_ms']} ms")
    print(f"p95:                 {results['p95_ms']} ms (Target: < 500 ms)")
    print(f"p99:                 {results['p99_ms']} ms")
    print(f"Max:                 {results['max_ms']} ms")
    print(f"Target Satisfied:    {results['under_500ms_target']}")

    # Save benchmark artifact
    reports_perf_dir = REPO_ROOT / "reports" / "metrics"
    reports_perf_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_perf_dir / "latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_latency_benchmark(100)
