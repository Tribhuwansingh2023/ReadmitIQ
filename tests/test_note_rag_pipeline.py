"""End-to-end dry run of the note-RAG pipeline on a small synthetic dataset.

Everything runs in a temporary directory through READMITIQ_*_DIR overrides, so no real data,
protected file or API is touched. Embeddings are local TF-IDF + SVD vectors (not Jina).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sqlite_vec")

ROOT = Path(__file__).resolve().parent.parent
SIZES = {"train": 1500, "val": 400, "test": 400}
GROUPS = ["circulatory", "diabetes", "respiratory", "digestive", "other"]


def synthetic_split(
    n: int, start: int, rng: np.random.Generator, feature_list: dict
) -> pd.DataFrame:
    from src.notes import generate_notes_for_dataframe

    d = pd.DataFrame(
        {
            "encounter_id": np.arange(start, start + n),
            "patient_nbr": rng.integers(start, start + n * 2 // 3, n),
            "age_num": rng.choice([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], n).astype(float),
            "time_in_hospital": rng.integers(1, 15, n),
        }
    )
    for col in feature_list["numeric_features"]:
        if col not in d.columns:
            d[col] = rng.integers(0, 6, n)
    for col in ("diag_1_group", "diag_2_group", "diag_3_group"):
        d[col] = rng.choice(GROUPS, n)
    choices = {
        "medical_specialty": ["InternalMedicine", "Cardiology", "?"],
        "payer_code": ["MC", "HM", "SP", "BC", "?"],
        "max_glu_serum": ["None", "Norm", ">200", ">300"],
        "A1Cresult": ["None", "Norm", ">7", ">8"],
        "discharge_disposition_group": ["home", "skilled_nursing"],
        "admission_type_group": ["emergency", "elective"],
        "admission_source_group": ["emergency_room", "referral"],
    }
    for col, values in choices.items():
        d[col] = rng.choice(values, n)
    for col in feature_list["categorical_features"]:
        if col not in d.columns:
            d[col] = rng.choice(["No", "Steady", "Up", "Down"], n)
    risk = 0.08 + 0.10 * (d["number_inpatient"] > 2)
    d["readmitted_30d"] = (rng.random(n) < risk).astype(int)
    d["clinical_note"] = generate_notes_for_dataframe(d)
    return d


def write_fake_embeddings(proc: Path) -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    sys.path.insert(0, str(ROOT / "scripts"))
    import embed_notes as en

    out = proc / "note_rag"
    for variant, col, path in (
        ("v1", "clinical_note", lambda s: proc / f"{s}.parquet"),
        ("v2", "clinical_note_v2", lambda s: out / f"notes_v2_{s}.parquet"),
    ):
        frames = {s: pd.read_parquet(path(s)) for s in SIZES}
        tfidf = TfidfVectorizer(min_df=2).fit(frames["train"][col])
        svd = TruncatedSVD(16, random_state=0).fit(tfidf.transform(frames["train"][col]))
        for split, df in frames.items():
            vecs = svd.transform(tfidf.transform(df[col]))
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
            name = en.embedding_file("minilm", variant, split)
            np.save(out / name, vecs.astype(np.float32))
            en.update_index(out, "minilm", variant, split, df["encounter_id"].to_numpy())
            manifest = en.read_manifest(out)
            manifest["entries"][name] = {
                "rows": len(vecs),
                "dim": vecs.shape[1],
                "sha256": en.sha256_of(out / name),
                "model_name": "test-tfidf-svd",
            }
            en.write_manifest(out, manifest)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("noterag")
    proc, reports, models = base / "processed", base / "reports", base / "models"
    for folder in (proc, reports, models):
        folder.mkdir()
    feature_list = json.loads(
        (ROOT / "data" / "processed" / "feature_list.json").read_text("utf-8")
    )
    rng = np.random.default_rng(11)
    start, meta = 100_000, {"splits": {}}
    for split, n in SIZES.items():
        synthetic_split(n, start, rng, feature_list).to_parquet(
            proc / f"{split}.parquet", index=False
        )
        meta["splits"][split] = {"rows": n}
        start += n * 3
    (proc / "feature_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    env = {
        **os.environ,
        "READMITIQ_PROCESSED_DIR": str(proc),
        "READMITIQ_REPORTS_DIR": str(reports),
        "READMITIQ_MODELS_DIR": str(models),
        "PYTHONIOENCODING": "utf-8",
    }

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
        )

    steps = run("-m", "src.notes_v2")
    assert steps.returncode == 0, steps.stdout + steps.stderr
    write_fake_embeddings(proc)
    return {"proc": proc, "reports": reports, "run": run}


def test_pipeline_end_to_end(workspace):
    run, reports, proc = workspace["run"], workspace["reports"], workspace["proc"]
    for script in ("select_note_retriever.py", "build_note_rag_features.py"):
        done = run(f"scripts/{script}")
        assert done.returncode == 0, done.stdout + done.stderr

    checks = json.loads((reports / "note_rag" / "note_rag_assertions.json").read_text("utf-8"))
    assert checks["all_passed"] and checks["test_labels_read"] is False
    assert checks["brute_force"]["neighbors_match_exactly"]
    assert checks["sqlite_vec_cross_check"]["passed"]

    for split, n in SIZES.items():
        feats = pd.read_parquet(proc / "note_rag" / f"note_rag_features_{split}.parquet")
        assert len(feats) == n and "readmitted_30d" not in feats.columns
        assert not feats.isna().any().any()

    done = run("scripts/prepare_colab_upload_noterag.py")
    assert done.returncode == 0, done.stdout + done.stderr
    upload = pd.read_parquet(proc / "colab_upload_noterag" / "colab_val_noterag.parquet")
    assert list(upload.columns) == [
        "encounter_id",
        "clinical_note",
        "readmitted_30d",
        "rag_note_rate_k20",
    ]

    done = run("scripts/run_note_rag_ablation.py", "--dry-run")
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((reports / "metrics" / "ablation_note_rag.json").read_text("utf-8"))
    assert report["split"] == "validation" and report["test_split_touched"] is False
    assert len(report["results"]) == 9 and len(report["paired_differences"]) == 6

    done = run("scripts/build_six_model_grid.py")
    assert done.returncode == 0, done.stdout + done.stderr
    grid = json.loads((reports / "metrics" / "six_model_grid.json").read_text("utf-8"))
    assert {r["status"] for r in grid["rows"] if r["model"] == "Transformer"} == {"pending"}
