"""Snapshot and verify SHA-256 hashes of the files the note-RAG work must never change.

Usage:
    python scripts/check_protected_files.py snapshot   # write the baseline once
    python scripts/check_protected_files.py verify     # compare against the baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

BASELINE_PATH = REPORTS_DIR / "note_rag" / "PROTECTED_SHA256.json"


def protected_paths() -> List[Path]:
    """List every protected file that exists, plus the fixed names that may be absent."""
    fixed = [
        MODELS_DIR / "FROZEN_SELECTION.json",
        METRICS_DIR / "test_results.json",
        METRICS_DIR / "test_eval.lock",
        METRICS_DIR / "ablation_table.json",
        METRICS_DIR / "ablation_table.md",
    ]
    for split in ("train", "val", "test"):
        fixed.append(PROCESSED_DATA_DIR / f"{split}.parquet")
        fixed.append(PROCESSED_DATA_DIR / f"{split}_rag.parquet")
    final_dir = MODELS_DIR / "final"
    found = sorted(p for p in final_dir.glob("*") if p.is_file()) if final_dir.exists() else []
    return list(dict.fromkeys(fixed + found))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_all() -> Dict[str, str]:
    """Map repo-relative posix path to SHA-256, or to 'absent' when the file does not exist."""
    result: Dict[str, str] = {}
    for path in protected_paths():
        key = path.relative_to(BASE_DIR).as_posix()
        result[key] = sha256_of(path) if path.exists() else "absent"
    return result


def verify(baseline: Dict[str, str], current: Dict[str, str]) -> List[str]:
    """Return a list of human-readable differences (empty means unchanged)."""
    problems = []
    for key, expected in baseline.items():
        actual = current.get(key, "absent")
        if actual != expected:
            problems.append(f"{key}: baseline {expected[:12]} now {actual[:12]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "verify"])
    args = parser.parse_args()

    current = hash_all()
    if args.mode == "snapshot":
        if BASELINE_PATH.exists():
            print("Baseline already exists; refusing to overwrite it.")
            return 1
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        present = sum(v != "absent" for v in current.values())
        print(f"Baseline written: {present} files hashed, {len(current) - present} absent.")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    problems = verify(baseline, current)
    present = sum(v != "absent" for v in baseline.values())
    if problems:
        print("PROTECTED FILES CHANGED:")
        print("\n".join(problems))
        return 1
    print(f"Protected files unchanged: {present} hashed, {len(baseline) - present} absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
