import json
import time
from pathlib import Path

import pandas as pd

from api.service import get_service

svc = get_service()
test_df = pd.read_parquet("data/processed/test_rag.parquet").head(1000)
records = test_df.to_dict(orient="records")

for r in records:
    r.pop("rag_struct_rate_k20", None)
    r.pop("readmitted_30d", None)

print(f"Benchmarking batch scoring on {len(records)} records...")
# Warmup
_ = svc.score_batch(records[:10])

t0 = time.perf_counter()
results = svc.score_batch(records)
duration = time.perf_counter() - t0

ms_per_row = (duration / len(records)) * 1000.0
print(f"Scored {len(results)} rows in {duration:.4f} s ({ms_per_row:.3f} ms/row).")
assert duration < 2.0, f"Batch scoring exceeded 2.0s limit: {duration} s"

bench_path = Path("reports/metrics/latency_benchmark.json")
with open(bench_path, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

bench_data["batch_vectorized_benchmark"] = {
    "batch_size": len(records),
    "total_seconds": round(duration, 4),
    "ms_per_row": round(ms_per_row, 3),
    "throughput_rows_per_second": round(len(records) / duration, 1),
    "under_2s_target": bool(duration < 2.0),
    "measured_at_step": "Step 2.1 vectorized scoring benchmark"
}

with open(bench_path, "w", encoding="utf-8") as f:
    json.dump(bench_data, f, indent=2)

print(f"[SUCCESS] Updated {bench_path} with batch_vectorized_benchmark: {duration:.4f}s")
