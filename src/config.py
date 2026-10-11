import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CONFIG_YAML_PATH = BASE_DIR / "configs" / "config.yaml"


def load_yaml_config(config_path: Path = CONFIG_YAML_PATH) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file with fallback if pyyaml is missing."""
    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # Safe pure-Python fallback parser when pyyaml is not installed in the active kernel
            cfg: Dict[str, Any] = {}
            curr_section: Optional[str] = None
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.rstrip()
                        if not line_str or line_str.strip().startswith("#"):
                            continue
                        if not line_str.startswith(" ") and line_str.endswith(":"):
                            curr_section = line_str[:-1].strip()
                            cfg[curr_section] = {}
                        elif line_str.startswith("  ") and ":" in line_str and curr_section:
                            parts = line_str.strip().split(":", 1)
                            k = parts[0].strip()
                            v = parts[1].strip().strip("\"'")
                            if v.isdigit():
                                v_val: Any = int(v)
                            else:
                                try:
                                    v_val = float(v)
                                except ValueError:
                                    v_val = v
                            cfg[curr_section][k] = v_val
                return cfg
            except Exception:
                pass
        except Exception:
            pass
    return {}


YAML_CFG = load_yaml_config()

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = BASE_DIR / YAML_CFG.get("paths", {}).get("raw_data", "data/raw/diabetic_data.csv")
INTERIM_DATA_DIR = BASE_DIR / YAML_CFG.get("paths", {}).get("interim_data_dir", "data/interim")
PROCESSED_DATA_DIR = BASE_DIR / YAML_CFG.get("paths", {}).get(
    "processed_data_dir", "data/processed"
)

TRAIN_PARQUET = PROCESSED_DATA_DIR / "train.parquet"
VAL_PARQUET = PROCESSED_DATA_DIR / "val.parquet"
TEST_PARQUET = PROCESSED_DATA_DIR / "test.parquet"

TRAIN_RAG_PARQUET = PROCESSED_DATA_DIR / "train_rag.parquet"
VAL_RAG_PARQUET = PROCESSED_DATA_DIR / "val_rag.parquet"
TEST_RAG_PARQUET = PROCESSED_DATA_DIR / "test_rag.parquet"

FEATURE_METADATA_PATH = PROCESSED_DATA_DIR / "feature_metadata.json"
FEATURE_LIST_PATH = PROCESSED_DATA_DIR / "feature_list.json"

MODELS_DIR = Path(
    os.getenv(
        "READMITIQ_MODELS_DIR",
        str(BASE_DIR / YAML_CFG.get("paths", {}).get("models_dir", "models")),
    )
)
PREPROCESSOR_BASE_PATH = MODELS_DIR / "preprocessor_base.joblib"
PREPROCESSOR_RAG_PATH = MODELS_DIR / "preprocessor_rag.joblib"
REPORTS_DIR = Path(
    os.getenv(
        "READMITIQ_REPORTS_DIR",
        str(BASE_DIR / YAML_CFG.get("paths", {}).get("reports_dir", "reports")),
    )
)
METRICS_DIR = Path(os.getenv("READMITIQ_METRICS_DIR", str(REPORTS_DIR / "metrics")))
FIGURES_DIR = Path(os.getenv("READMITIQ_FIGURES_DIR", str(REPORTS_DIR / "figures")))
DOCS_DIR = BASE_DIR / "docs"

# Split & Seed
RANDOM_SEED = int(YAML_CFG.get("project", {}).get("random_seed", 42))
TRAIN_SPLIT_RATIO = float(YAML_CFG.get("split", {}).get("train_ratio", 0.70))
VAL_SPLIT_RATIO = float(YAML_CFG.get("split", {}).get("val_ratio", 0.15))
TEST_SPLIT_RATIO = float(YAML_CFG.get("split", {}).get("test_ratio", 0.15))
SPLIT_N_SPLITS = int(YAML_CFG.get("split", {}).get("n_splits", 10))

# Exclusions
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]  # Expired or hospice
DROP_COLS = ["weight", "examide", "citoglipton"]

ALL_DRUG_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

COUNT_COLS_TO_CAP = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "total_prior_visits",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
    "meds_per_day",
    "labs_per_day",
]

AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}
