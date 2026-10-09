#!/usr/bin/env python3
"""Screenshot verification and capture utility for ReadmitIQ presentation assets."""

import io
import sys
from pathlib import Path

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT_DIR / "docs" / "screenshots"


def verify_or_capture_screenshots():
    print("=" * 65)
    print("📸 ReadmitIQ Presentation Screenshot Verification")
    print("=" * 65)

    required_screenshots = [
        "cohort_triage_desktop.png",
        "cohort_triage_light_desktop.png",
        "patient_risk_desktop.png",
        "model_lab_desktop.png",
        "fairness_drift_desktop.png",
        "how_it_works_desktop.png",
        "cohort_triage_mobile.png",
        "patient_risk_mobile.png",
        "model_lab_mobile.png",
        "fairness_drift_mobile.png",
        "how_it_works_mobile.png",
    ]

    all_present = True
    for fname in required_screenshots:
        p = SCREENSHOTS_DIR / fname
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  ✓ {fname:<35} ({size_kb:>6.1f} KB)")
        else:
            print(f"  ✗ {fname:<35} [MISSING]")
            all_present = False

    print("-" * 65)
    if all_present:
        print("[SUCCESS] All required presentation screenshots are verified and present.")
    else:
        print("[NOTICE] Some screenshots missing. Ensure web dev server is running on :5173.")
    print("=" * 65 + "\n")
    return all_present


if __name__ == "__main__":
    success = verify_or_capture_screenshots()
    sys.exit(0 if success else 1)
