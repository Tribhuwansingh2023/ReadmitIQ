"""Clinical note template v2, used only for note-embedding retrieval.

v1 notes (src/notes.py) stay untouched and remain what the transformer reads. v2 is one
fixed template built from the cleaned structured columns. It contains no race, no label
words and no free text; every field is always present, using "not recorded" when empty.

Run `python -m src.notes_v2` to write data/processed/note_rag/notes_v2_{split}.parquet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.config import ALL_DRUG_COLS
from src.note_rag_config import (
    ID_COL,
    NOTE_RAG_DATA_DIR,
    SPLITS,
    expected_split_rows,
    notes_v2_path,
    split_parquet,
)

NOTE_COL = "clinical_note_v2"
NOT_RECORDED = "not recorded"
MINOR_AGE_LIMIT = 18

# Fixed labels, in the order they appear in every note. Used to check template drift.
TEMPLATE_LABELS: Sequence[str] = (
    "Discharge summary.",
    "Patient:",
    "Days in hospital:",
    "Prior year visits:",
    "Number of diagnoses:",
    "Diagnosis groups:",
    "Admitting specialty:",
    "Payer category:",
    "Glucose serum:",
    "A1C:",
    "Active diabetes drugs:",
    "Drugs with dose changes:",
    "Discharge destination:",
    "Admission type:",
    "Admission source:",
)

REQUIRED_COLUMNS: Sequence[str] = (
    "age_num",
    "time_in_hospital",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "number_diagnoses",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "medical_specialty",
    "payer_code",
    "max_glu_serum",
    "A1Cresult",
    "discharge_disposition_group",
    "admission_type_group",
    "admission_source_group",
)

FORBIDDEN_LABEL_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"\breadmit\b", re.IGNORECASE),
    re.compile(r"\breadmission\b", re.IGNORECASE),
    re.compile(r"\breadmitted\b", re.IGNORECASE),
    re.compile(r"(?<!\d)<30(?!\d)"),
    re.compile(r"(?<!\d)>30(?!\d)"),
    re.compile(r"\boutcome\b", re.IGNORECASE),
    re.compile(r"\brecurrent\b", re.IGNORECASE),
    re.compile(r"\brace\b", re.IGNORECASE),
)

_MISSING_TOKENS = {"", "?", "nan", "none", "unknown", "null", "n/a"}
_PAYER_CATEGORIES: Dict[str, str] = {
    "MC": "Medicare",
    "MD": "Medicaid",
    "HM": "health maintenance organization",
    "SP": "self-pay",
}
_GLUCOSE_TEXT: Dict[str, str] = {
    "none": "not tested",
    "norm": "normal",
    ">200": "above 200 mg/dL",
    ">300": "above 300 mg/dL",
}
_A1C_TEXT: Dict[str, str] = {
    "none": "not tested",
    "norm": "normal",
    ">7": "above 7 percent",
    ">8": "above 8 percent",
}
_DOSE_DIRECTION: Dict[str, str] = {"Up": "increased", "Down": "decreased"}


def plural(count: int, word: str) -> str:
    """Return '1 word' or 'n words'."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _MISSING_TOKENS


def clean_label(value: Any) -> str:
    """Make a categorical value readable: underscores to spaces, CamelCase split, lowercase."""
    if is_missing(value):
        return NOT_RECORDED
    text = str(value).strip().replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return re.sub(r"\s+", " ", text).lower()


def age_phrase(age_num: Any) -> str:
    """Age as a decade; minors are never called adults."""
    if is_missing(age_num):
        return f"age {NOT_RECORDED}"
    age = int(age_num)
    if age < MINOR_AGE_LIMIT:
        return "child or adolescent under 18"
    return f"adult in their {(age // 10) * 10}s"


def payer_phrase(payer_code: Any) -> str:
    if is_missing(payer_code):
        return NOT_RECORDED
    code = str(payer_code).strip()
    if code.upper() in _PAYER_CATEGORIES:
        return _PAYER_CATEGORIES[code.upper()]
    if len(code) > 3:
        return clean_label(code)
    return "other insurance"


def lab_phrase(value: Any, table: Dict[str, str]) -> str:
    if is_missing(value):
        return table["none"]
    return table.get(str(value).strip().lower(), NOT_RECORDED)


def count_active_drugs(row: Dict[str, Any], drug_cols: Sequence[str]) -> int:
    """Use the cleaned n_active_drugs column when present, else count non-'No' drug columns."""
    if not is_missing(row.get("n_active_drugs")):
        return int(row["n_active_drugs"])
    return sum(str(row.get(d, "No")).strip() in ("Steady", "Up", "Down") for d in drug_cols)


def dose_change_phrase(row: Dict[str, Any], drug_cols: Sequence[str]) -> str:
    changes: List[str] = []
    for drug in drug_cols:
        direction = _DOSE_DIRECTION.get(str(row.get(drug, "No")).strip())
        if direction:
            changes.append(f"{drug} {direction}")
    return ", ".join(changes) if changes else "none"


def _int(value: Any) -> int:
    return 0 if is_missing(value) else int(value)


def build_note(row: Dict[str, Any], drug_cols: Sequence[str]) -> str:
    """Build one v2 note from a dict of cleaned structured fields."""
    days = _int(row.get("time_in_hospital"))
    n_active = count_active_drugs(row, drug_cols)
    parts = [
        "Discharge summary.",
        f"Patient: {age_phrase(row.get('age_num'))}.",
        f"Days in hospital: {days}.",
        "Prior year visits: "
        f"{plural(_int(row.get('number_inpatient')), 'inpatient visit')}, "
        f"{plural(_int(row.get('number_emergency')), 'emergency visit')}, "
        f"{plural(_int(row.get('number_outpatient')), 'outpatient visit')}.",
        f"Number of diagnoses: {_int(row.get('number_diagnoses'))}.",
        "Diagnosis groups: "
        f"primary {clean_label(row.get('diag_1_group'))}, "
        f"secondary {clean_label(row.get('diag_2_group'))}, "
        f"tertiary {clean_label(row.get('diag_3_group'))}.",
        f"Admitting specialty: {clean_label(row.get('medical_specialty'))}.",
        f"Payer category: {payer_phrase(row.get('payer_code'))}.",
        f"Glucose serum: {lab_phrase(row.get('max_glu_serum'), _GLUCOSE_TEXT)}.",
        f"A1C: {lab_phrase(row.get('A1Cresult'), _A1C_TEXT)}.",
        f"Active diabetes drugs: {n_active}.",
        f"Drugs with dose changes: {dose_change_phrase(row, drug_cols)}.",
        f"Discharge destination: {clean_label(row.get('discharge_disposition_group'))}.",
        f"Admission type: {clean_label(row.get('admission_type_group'))}.",
        f"Admission source: {clean_label(row.get('admission_source_group'))}.",
    ]
    return " ".join(parts)


def drug_columns_in(df: pd.DataFrame) -> List[str]:
    return [c for c in ALL_DRUG_COLS if c in df.columns]


def build_notes(df: pd.DataFrame) -> List[str]:
    """Build v2 notes for every row of a cleaned split dataframe, preserving row order."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build v2 notes, missing columns: {missing}")
    drug_cols = drug_columns_in(df)
    wanted = list(dict.fromkeys(list(REQUIRED_COLUMNS) + ["n_active_drugs"] + drug_cols))
    records = df[[c for c in wanted if c in df.columns]].to_dict(orient="records")
    return [build_note(r, drug_cols) for r in records]


def label_word_hits(notes: Iterable[str]) -> Dict[str, int]:
    """Count notes matching each forbidden label pattern (all values must be zero)."""
    hits = {p.pattern: 0 for p in FORBIDDEN_LABEL_PATTERNS}
    for note in notes:
        for pattern in FORBIDDEN_LABEL_PATTERNS:
            if pattern.search(note):
                hits[pattern.pattern] += 1
    return hits


def template_violations(notes: Iterable[str]) -> int:
    """Count notes whose fixed labels are missing, repeated or out of order."""
    bad = 0
    for note in notes:
        positions = [note.find(label) for label in TEMPLATE_LABELS]
        in_order = all(p >= 0 for p in positions) and positions == sorted(positions)
        repeated = any(note.count(label) != 1 for label in TEMPLATE_LABELS)
        bad += int(not in_order or repeated)
    return bad


def minors_called_adult(df: pd.DataFrame, notes: Sequence[str]) -> int:
    """Count notes written for a patient under 18 that contain the word 'adult'."""
    minor = (df["age_num"].astype(float) < MINOR_AGE_LIMIT).to_numpy()
    return sum(bool(re.search(r"\badult\b", note, re.I)) for note, m in zip(notes, minor) if m)


def validate_notes(df: pd.DataFrame, notes: Sequence[str]) -> Dict[str, Any]:
    """Run every v2 note check on all rows and return the measured counts."""
    if len(df) != len(notes):
        raise ValueError(f"Row count mismatch: {len(df)} rows but {len(notes)} notes")
    return {
        "rows": len(notes),
        "label_word_hits": label_word_hits(notes),
        "template_violations": template_violations(notes),
        "minors_called_adult": minors_called_adult(df, notes),
        "empty_notes": sum(not n.strip() for n in notes),
        "unique_notes": len(set(notes)),
    }


def assert_valid(report: Dict[str, Any], split: str) -> None:
    if any(report["label_word_hits"].values()):
        raise AssertionError(
            f"{split}: label or race words in v2 notes: {report['label_word_hits']}"
        )
    for key in ("template_violations", "minors_called_adult", "empty_notes"):
        if report[key]:
            raise AssertionError(f"{split}: {key} = {report[key]}")


def write_notes_v2(
    split: str,
    out_dir: Path = NOTE_RAG_DATA_DIR,
    expected_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Build, validate and save notes_v2_{split}.parquet from the cleaned split parquet."""
    df = pd.read_parquet(split_parquet(split))
    if ID_COL not in df.columns:
        raise ValueError(f"{split}_parquet has no {ID_COL} column")
    if expected_rows is not None and len(df) != expected_rows:
        raise AssertionError(f"{split}: {len(df)} rows, expected {expected_rows}")
    notes = build_notes(df)
    report = validate_notes(df, notes)
    assert_valid(report, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), NOTE_COL: notes})
    out.to_parquet(notes_v2_path(split, out_dir), index=False)
    return report


def main() -> None:
    expected = expected_split_rows()
    for split in SPLITS:
        report = write_notes_v2(split, expected_rows=expected[split])
        print(
            f"[{split}] wrote {report['rows']} notes, "
            f"{report['unique_notes']} unique, all checks passed"
        )


if __name__ == "__main__":
    main()
