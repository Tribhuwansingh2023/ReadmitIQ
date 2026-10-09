"""Pytest configuration, environment guards, and fixtures strategy for ReadmitIQ."""

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def has_processed_data() -> bool:
    """Return True if processed dataset parquets are present locally."""
    tr = ROOT_DIR / "data" / "processed" / "train.parquet"
    va = ROOT_DIR / "data" / "processed" / "val.parquet"
    te = ROOT_DIR / "data" / "processed" / "test.parquet"
    return tr.exists() and va.exists() and te.exists()


def has_full_retriever_artifacts() -> bool:
    """Return True if the large 17MB FAISS index is present locally."""
    faiss_p = ROOT_DIR / "models" / "final" / "retriever_faiss.index"
    meta_p = ROOT_DIR / "models" / "final" / "retriever_meta.parquet"
    return faiss_p.exists() and meta_p.exists()


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Ensure tests run cleanly even in CI environments without large data/model artifacts.

    If full models are absent, falls back to lightweight fixtures in tests/fixtures/.
    """
    models_dir = ROOT_DIR / "models"
    final_model = models_dir / "final" / "model.joblib"

    if not final_model.exists():
        fixture_models = ROOT_DIR / "tests" / "fixtures" / "models"
        fixture_reports = ROOT_DIR / "tests" / "fixtures" / "reports"
        if fixture_models.exists():
            os.environ["READMITIQ_MODELS_DIR"] = str(fixture_models)
            os.environ["READMITIQ_REPORTS_DIR"] = str(fixture_reports)
            os.environ["READMITIQ_SERVE_MODE"] = "lite"
            print("\n[CI Environment] Utilizing lightweight tests/fixtures/ for test execution.")
