"""Tests for the note-embedding RAG feature. All fixtures are tiny and synthetic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from src import note_rag as nr
from src import notes_v2
from src.note_rag_config import FEATURE_NAMES, K_GRID, PRIMARY_FEATURE

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
AGES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
GROUPS = ["circulatory", "diabetes", "respiratory", "other"]


def make_cleaned_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "encounter_id": np.arange(1000, 1000 + n),
            "patient_nbr": rng.integers(0, max(n // 2, 1), n),
            "age_num": rng.choice(AGES, n),
            "time_in_hospital": rng.integers(1, 15, n),
            "number_inpatient": rng.integers(0, 4, n),
            "number_emergency": rng.integers(0, 3, n),
            "number_outpatient": rng.integers(0, 3, n),
            "number_diagnoses": rng.integers(1, 10, n),
            "diag_1_group": rng.choice(GROUPS, n),
            "diag_2_group": rng.choice(GROUPS, n),
            "diag_3_group": rng.choice(GROUPS, n),
            "medical_specialty": rng.choice(["InternalMedicine", "Cardiology", "?", None], n),
            "payer_code": rng.choice(["MC", "HM", "SP", "BC", "?", None], n),
            "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"], n),
            "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n),
            "discharge_disposition_group": rng.choice(["home", "skilled_nursing"], n),
            "admission_type_group": rng.choice(["emergency", "elective"], n),
            "admission_source_group": rng.choice(["emergency_room", "referral"], n),
            "n_active_drugs": rng.integers(0, 4, n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "metformin": rng.choice(["No", "Steady", "Up", "Down"], n),
        }
    )


def make_toy(
    n: int = 240, dim: int = 8, seed: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clustered unit vectors, patients with several encounters, cluster-linked labels."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(6, dim))
    cluster = rng.integers(0, 6, n)
    vecs = centers[cluster] + 0.4 * rng.normal(size=(n, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    patients = rng.integers(0, n // 3, n)
    y = (rng.random(n) < 0.1 + 0.15 * (cluster % 2)).astype(int)
    return vecs.astype(np.float32), patients.astype(np.int64), y


# --------------------------------------------------------------------------------------
# Notes v2
# --------------------------------------------------------------------------------------
def test_notes_v2_all_checks_pass():
    df = make_cleaned_frame()
    notes = notes_v2.build_notes(df)
    report = notes_v2.validate_notes(df, notes)
    notes_v2.assert_valid(report, "toy")
    assert report["rows"] == len(df)


def test_notes_v2_every_field_present_in_every_note():
    df = make_cleaned_frame()
    for note in notes_v2.build_notes(df):
        for label in notes_v2.TEMPLATE_LABELS:
            assert note.count(label) == 1


def test_notes_v2_no_adult_for_minors_and_age_decade():
    df = make_cleaned_frame(n=200)
    notes = notes_v2.build_notes(df)
    for age, note in zip(df["age_num"], notes):
        if age < 18:
            assert "adult" not in note.lower()
            assert "child or adolescent" in note
        else:
            assert f"in their {(age // 10) * 10}s" in note


def test_notes_v2_plurals_and_missing_values():
    row = make_cleaned_frame(1).iloc[0].to_dict()
    row.update(
        number_inpatient=1,
        number_emergency=2,
        medical_specialty="?",
        payer_code=None,
        insulin="Up",
        metformin="Down",
    )
    note = notes_v2.build_note(row, ["insulin", "metformin"])
    assert "1 inpatient visit," in note and "2 emergency visits" in note
    assert "Admitting specialty: not recorded." in note
    assert "Payer category: not recorded." in note
    assert "insulin increased, metformin decreased" in note


def test_notes_v2_have_no_label_or_race_words():
    df = make_cleaned_frame()
    hits = notes_v2.label_word_hits(notes_v2.build_notes(df))
    assert not any(hits.values())


def test_notes_v2_detects_template_drift_and_label_words():
    df = make_cleaned_frame(3)
    notes = notes_v2.build_notes(df)
    assert notes_v2.template_violations([notes[0].replace("A1C:", "HbA1c:")]) == 1
    assert any(notes_v2.label_word_hits([notes[0] + " Patient was readmitted."]).values())


def test_notes_v2_missing_column_raises():
    with pytest.raises(ValueError):
        notes_v2.build_notes(make_cleaned_frame().drop(columns=["A1Cresult"]))


# --------------------------------------------------------------------------------------
# Embedding validation
# --------------------------------------------------------------------------------------
def write_embeddings(
    out: Path, vecs: np.ndarray, ids: np.ndarray, split: str = "train"
) -> Dict[str, str]:
    name = f"embeddings_minilm_v2_{split}.npy"
    np.save(out / name, vecs)
    index = pd.DataFrame({"encounter_id": ids, "split": split, "row": np.arange(len(ids))})
    index.to_parquet(out / "embeddings_minilm_v2_index.parquet", index=False)
    sha = hashlib.sha256((out / name).read_bytes()).hexdigest()
    manifest = {"entries": {name: {"rows": len(ids), "dim": vecs.shape[1], "sha256": sha}}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"name": name}


def test_load_embeddings_roundtrip(tmp_path):
    vecs, _, _ = make_toy(30)
    ids = np.arange(30) + 5
    write_embeddings(tmp_path, vecs, ids)
    loaded = nr.load_embeddings(tmp_path, "minilm", "v2", "train", ids)
    assert np.array_equal(loaded, vecs)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("rows", "rows but the parquet has"),
        ("nan", "NaN"),
        ("norm", "unit norm"),
        ("dtype", "float32"),
        ("order", "encounter order"),
        ("sha", "SHA-256"),
    ],
)
def test_load_embeddings_rejects_bad_files(tmp_path, mutation, message):
    vecs, _, _ = make_toy(30)
    ids = np.arange(30) + 5
    expected_ids = ids.copy()
    if mutation == "rows":
        vecs = vecs[:-1]
        ids = ids[:-1]
        expected_ids = np.arange(30) + 5
    elif mutation == "nan":
        vecs = vecs.copy()
        vecs[3, 0] = np.nan
    elif mutation == "norm":
        vecs = vecs * 2
    elif mutation == "dtype":
        vecs = vecs.astype(np.float64)
    elif mutation == "order":
        expected_ids = ids[::-1].copy()
    write_embeddings(tmp_path, vecs, ids)
    if mutation == "sha":
        np.save(tmp_path / "embeddings_minilm_v2_train.npy", vecs * 0.5)
    with pytest.raises(nr.EmbeddingValidationError, match=message):
        nr.load_embeddings(tmp_path, "minilm", "v2", "train", expected_ids)


def test_load_embeddings_missing_manifest_entry(tmp_path):
    vecs, _, _ = make_toy(10)
    ids = np.arange(10)
    write_embeddings(tmp_path, vecs, ids)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"entries": {}}), encoding="utf-8")
    with pytest.raises(nr.EmbeddingValidationError, match="not listed"):
        nr.load_embeddings(tmp_path, "minilm", "v2", "train", ids)


# --------------------------------------------------------------------------------------
# Search, leakage and brute force
# --------------------------------------------------------------------------------------
def test_search_never_returns_same_patient_and_never_self():
    vecs, patients, _ = make_toy()
    idx, _ = nr.exact_search(vecs, vecs, patients, patients, k=10)
    assert nr.assert_no_same_patient(idx, patients, patients) == len(vecs)
    assert not (idx == np.arange(len(vecs))[:, None]).any()


def test_assert_no_same_patient_detects_leak():
    vecs, patients, _ = make_toy(60)
    idx, _ = nr.exact_search(vecs, vecs, patients, patients, k=5)
    idx[0, 0] = np.flatnonzero(patients == patients[0])[0]
    with pytest.raises(nr.LeakageError):
        nr.assert_no_same_patient(idx, patients, patients)


def test_exact_search_matches_naive_ordering():
    vecs, patients, _ = make_toy(80)
    idx, sims = nr.exact_search(vecs[:10], vecs, patients[:10], patients, k=7)
    for row in range(10):
        order, _ = nr.brute_force_row(
            vecs[row], patients[row], vecs, patients, np.zeros(len(vecs)), 0.0, (7,)
        )
        assert np.array_equal(idx[row], order[:7])
        assert np.allclose(sims[row], vecs[order[:7]].astype(float) @ vecs[row].astype(float))


def test_ties_are_broken_by_index_position():
    base = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (6, 1))
    patients = np.arange(6, dtype=np.int64)
    idx, _ = nr.exact_search(base[:1], base, patients[:1], patients, k=3)
    assert idx[0].tolist() == [1, 2, 3]


def test_fewer_eligible_neighbors_than_k_pads_with_minus_one():
    vecs, _, _ = make_toy(5)
    patients = np.array([0, 0, 1, 1, 2])
    idx, sims = nr.exact_search(vecs[:1], vecs, patients[:1], patients, k=4)
    assert (idx[0] >= 0).sum() == 3 and (idx[0, 3] == -1) and np.isnan(sims[0, 3])


def test_folds_are_patient_disjoint_and_partition_rows():
    vecs, patients, y = make_toy()
    folds = nr.make_folds(y, patients)
    nr.assert_folds_patient_disjoint(folds, patients, len(y))


def test_fold_check_detects_shared_patient():
    _, patients, y = make_toy()
    folds = nr.make_folds(y, patients)
    idx_in, idx_out = folds[0]
    bad = [(np.append(idx_in, idx_out[0]), idx_out)] + list(folds[1:])
    with pytest.raises(nr.LeakageError):
        nr.assert_folds_patient_disjoint(bad, patients, len(y))


def test_oof_equals_brute_force_on_toy_set():
    vecs, patients, y = make_toy()
    result = nr.build_train_features(vecs, patients, y)
    assert nr.assert_no_same_patient(result.idx, patients, patients) == len(y)
    nr.assert_full_neighbors(result.idx, max(K_GRID))
    report = nr.check_oof_against_brute_force(
        vecs, patients, y, result.folds, result.idx, result.frame, rows=range(len(y))
    )
    assert report["neighbors_match_exactly"]


def test_oof_never_uses_own_label():
    vecs, patients, y = make_toy()
    flipped = 1 - y
    a = nr.build_train_features(vecs, patients, y)
    b = nr.build_train_features(vecs, patients, flipped)
    # Neighbors are identical; only neighbor labels feed the feature. Flipping every label
    # flips every neighbor label, so rate_k20 must equal 1 - original, never depend on self.
    assert np.array_equal(a.idx, b.idx)
    assert np.allclose(a.frame[PRIMARY_FEATURE] + b.frame[PRIMARY_FEATURE], 1.0, atol=1e-6)


def test_val_queries_use_train_index_only():
    vecs, patients, y = make_toy(300)
    train, val = slice(0, 200), slice(200, 300)
    idx, sims, frame = nr.build_query_features(
        vecs[val], patients[val], vecs[train], patients[train], y[train]
    )
    assert idx.max() < 200
    nr.assert_no_same_patient(idx, patients[val], patients[train])
    nr.assert_features_valid(frame, n_neighbors_full=True)


def test_features_frame_columns_and_grid_values():
    vecs, patients, y = make_toy()
    frame = nr.build_train_features(vecs, patients, y).frame
    assert list(frame.columns) == list(FEATURE_NAMES)
    nr.assert_features_valid(frame, n_neighbors_full=True)


def test_feature_validation_rejects_nan_and_off_grid():
    vecs, patients, y = make_toy()
    frame = nr.build_train_features(vecs, patients, y).frame
    bad = frame.copy()
    bad.loc[0, PRIMARY_FEATURE] = np.nan
    with pytest.raises(AssertionError):
        nr.assert_features_valid(bad, True)
    off = frame.copy()
    off.loc[0, PRIMARY_FEATURE] = 0.123
    with pytest.raises(AssertionError):
        nr.assert_features_valid(off, True)


def test_same_k_grid_check():
    nr.assert_same_k_grid({"train": K_GRID, "val": K_GRID, "test": K_GRID})
    with pytest.raises(nr.LeakageError):
        nr.assert_same_k_grid({"train": K_GRID, "val": (5, 10, 20)})


def test_shuffled_label_control_is_near_chance_and_real_signal_is_not():
    rng = np.random.default_rng(3)
    n, dim = 3000, 16
    centers = rng.normal(size=(8, dim))
    cluster = rng.integers(0, 8, n)
    vecs = centers[cluster] + 0.3 * rng.normal(size=(n, dim))
    vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
    patients = np.arange(n, dtype=np.int64)
    y = (rng.random(n) < 0.05 + 0.4 * (cluster == 0)).astype(int)
    train, val = slice(0, 2000), slice(2000, n)
    idx, sims, frame = nr.build_query_features(
        vecs[val], patients[val], vecs[train], patients[train], y[train]
    )
    real = nr.univariate_auc(y[val], frame[PRIMARY_FEATURE].to_numpy())
    control = nr.shuffled_label_control(idx, sims, y[train], y[val])
    assert real > 0.7
    assert abs(control - 0.5) < 0.06


def test_fallback_prior_used_when_no_neighbors():
    idx = np.full((2, 50), -1)
    sims = np.full((2, 50), np.nan)
    frame = nr.features_from_neighbors(idx, sims, np.array([0, 1]), prior=0.25)
    assert np.allclose(frame[PRIMARY_FEATURE], 0.25)
    assert (frame["rag_note_sim_top1"] == 0).all()


# --------------------------------------------------------------------------------------
# SQLite store
# --------------------------------------------------------------------------------------
def test_sqlite_store_matches_exact_search(tmp_path):
    try:
        nr.open_sqlite_store(tmp_path / "probe.sqlite").close()
    except nr.SqliteStoreUnavailable as exc:
        pytest.skip(str(exc))
    vecs, patients, y = make_toy(300, dim=12)
    train, val = slice(0, 220), slice(220, 300)
    conn = nr.build_sqlite_store(
        tmp_path / "store.sqlite", vecs[train], np.arange(220), patients[train], y[train]
    )
    idx, sims = nr.exact_search(vecs[val], vecs[train], patients[val], patients[train], 20)
    report = nr.sqlite_recall_at_k(
        conn, vecs[train], vecs[val], patients[val], idx, sims, n_queries=40
    )
    conn.close()
    assert report["tie_aware_recall_at_k"] >= 0.99


# --------------------------------------------------------------------------------------
# Ablation helpers, grid builder and IO guards
# --------------------------------------------------------------------------------------
def fake_predictions(path: Path, n: int = 400, seed: int = 0, signal: float = 0.0) -> None:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    prob = np.clip(rng.random(n) * 0.6 + signal * y, 0, 1)
    pd.DataFrame({"encounter_id": np.arange(n), "readmitted_30d": y, "prob": prob}).to_csv(
        path, index=False
    )


def fake_note_report(path: Path) -> None:
    from src.note_rag_ablation import metric_row, paired_difference

    rng = np.random.default_rng(5)
    y = (rng.random(300) < 0.2).astype(int)
    p0, p1 = rng.random(300), rng.random(300)
    results, paired = [], []
    for model in ("Logistic Regression", "Random Forest", "LightGBM"):
        results.append(metric_row(y, p0, model, "none", None))
        results.append(metric_row(y, p1, model, "A", "note embeddings v2 / minilm"))
        paired.append({**paired_difference(y, p0, p1), "model": model, "feature_set": "A"})
    report = {"results": results, "paired_differences": paired, "retriever_label": "toy"}
    path.write_text(json.dumps(report), encoding="utf-8")


def test_grid_marks_missing_artifacts_pending(tmp_path):
    from src.six_model_grid import build_grid, render_markdown

    grid = build_grid(tmp_path / "none.json", tmp_path / "a.csv", tmp_path / "b.csv", tmp_path)
    assert len(grid["rows"]) == 8
    assert all(r["status"] == "pending" for r in grid["rows"])
    assert "pending" in render_markdown(grid)
    assert str(tmp_path) not in json.dumps(grid)


def test_grid_transformer_noterag_pending_until_file_exists(tmp_path):
    from src.six_model_grid import build_grid

    note_json = tmp_path / "ablation_note_rag.json"
    fake_note_report(note_json)
    fake_predictions(tmp_path / "without.csv")
    grid = build_grid(note_json, tmp_path / "without.csv", tmp_path / "with.csv", tmp_path)
    transformer = {r["rag"]: r for r in grid["rows"] if r["model"] == "Transformer"}
    assert transformer["without RAG"]["status"] == "ok"
    assert transformer["with note-RAG"]["status"] == "pending"

    fake_predictions(tmp_path / "with.csv", signal=0.3)
    grid = build_grid(note_json, tmp_path / "without.csv", tmp_path / "with.csv", tmp_path)
    transformer = {r["rag"]: r for r in grid["rows"] if r["model"] == "Transformer"}
    assert transformer["with note-RAG"]["status"] == "ok"
    assert transformer["with note-RAG"]["paired_delta_auc"]["observed_diff"] > 0
    lightgbm = [r for r in grid["rows"] if r["model"] == "LightGBM"]
    assert {r["role"] for r in lightgbm} == {"extra"}
    assert grid["rows"][-1]["model"] == "LightGBM"


def test_grid_rejects_misaligned_transformer_files(tmp_path):
    from src.six_model_grid import build_grid

    fake_predictions(tmp_path / "a.csv")
    fake_predictions(tmp_path / "b.csv")
    other = pd.read_csv(tmp_path / "b.csv")
    other["encounter_id"] += 1
    other.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        build_grid(tmp_path / "none.json", tmp_path / "a.csv", tmp_path / "b.csv", tmp_path)


def test_verdict_requires_ci_excluding_zero():
    from src.note_rag_ablation import summarize_verdicts, verdict_from_ci

    assert verdict_from_ci(0.001, 0.01) == "measurable gain"
    assert verdict_from_ci(-0.002, 0.01) == "no measurable predictive lift"
    assert verdict_from_ci(-0.02, -0.001) == "measurable loss"
    none = [{"verdict": "no measurable predictive lift", "model": "x", "feature_set": "A"}]
    assert summarize_verdicts(none).startswith("No measurable predictive lift")
    lost = none + [{"verdict": "measurable loss", "model": "RF", "feature_set": "B"}]
    assert "measurable loss" in summarize_verdicts(lost) and "RF (set B)" in summarize_verdicts(
        lost
    )
    won = none + [{"verdict": "measurable gain", "model": "LR", "feature_set": "A"}]
    assert summarize_verdicts(won) == "Measurable gain for: LR (set A)"


def test_reproduction_check_passes_and_stops():
    from src.note_rag_ablation import ReproductionError, check_reproduction

    row = {
        "model": "LightGBM",
        "rag_feature": False,
        "auc_roc": 0.6722,
        "f1": 0.28,
        "precision_top_10pct": 0.25,
    }
    ours = {**row, "precision_top_10pct": 0.25}
    assert check_reproduction([ours], [row])["passed"]
    drift = {**row, "auc_roc": 0.6722 + 0.002}
    with pytest.raises(ReproductionError):
        check_reproduction([drift], [row])


def test_test_split_label_can_never_be_loaded():
    from src.note_rag_io import load_split_meta

    with pytest.raises(PermissionError):
        load_split_meta("test", columns=["readmitted_30d"])


def test_protected_file_checker_detects_change(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_protected_files",
        Path(__file__).resolve().parent.parent / "scripts" / "check_protected_files.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = {"a": "x", "b": "absent"}
    assert module.verify(base, dict(base)) == []
    assert module.verify(base, {"a": "y", "b": "absent"})


# --------------------------------------------------------------------------------------
# Jina client and cached embedding script (no network: the API is faked)
# --------------------------------------------------------------------------------------
def fake_post_factory(dim: int = 8):
    calls = {"requests": 0, "texts": 0}

    def fake_post(url, payload, api_key):
        calls["requests"] += 1
        texts = payload["input"]
        calls["texts"] += len(texts)
        data = []
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vec = np.random.default_rng(seed).normal(size=dim)
            data.append({"index": i, "embedding": vec.tolist()})
        data.reverse()  # the client must restore input order
        return {"data": data, "usage": {"total_tokens": 10 * len(texts)}}

    return fake_post, calls


def test_jina_embed_texts_orders_normalizes_and_counts_tokens():
    from src import jina_client

    cfg = jina_client.JinaConfig("secret-key", "m", "r")
    post, _ = fake_post_factory()
    vecs, tokens = jina_client.embed_texts(["a", "b", "c"], cfg, post)
    again, _ = jina_client.embed_texts(["c"], cfg, post)
    assert vecs.dtype == np.float32 and tokens == 30
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
    assert np.allclose(vecs[2], again[0], atol=1e-6)
    assert "secret-key" not in repr(cfg)


def test_jina_post_retries_rate_limit_then_succeeds_and_hides_key():
    import io
    import urllib.error

    from src import jina_client

    attempts = {"n": 0}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "slow down", {}, None)
        return FakeResponse(b'{"ok": true}')

    result = jina_client.post_json(
        "https://x/y", {}, "secret-key", opener=opener, sleep=lambda s: None
    )
    assert result == {"ok": True} and attempts["n"] == 3

    def always_bad(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, None)

    with pytest.raises(jina_client.JinaError) as info:
        jina_client.post_json(
            "https://x/y", {}, "secret-key", opener=always_bad, sleep=lambda s: None
        )
    assert "secret-key" not in str(info.value)


def test_jina_rerank_parses_results():
    from src import jina_client

    cfg = jina_client.JinaConfig("k", "m", "r")

    def post(url, payload, key):
        assert url.endswith("/rerank") and payload["top_n"] == 2
        return {
            "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        }

    assert jina_client.rerank("q", ["a", "b"], cfg, top_n=2, post=post) == [(1, 0.9), (0, 0.1)]


def test_jina_script_caches_dedups_and_resumes(tmp_path, monkeypatch):
    import importlib.util
    import sys

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "embed_notes_jina", scripts / "embed_notes_jina.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["embed_notes_jina"] = module
    spec.loader.exec_module(module)
    from src import jina_client

    notes = ["note one", "note two", "note one", "note three"]
    pd.DataFrame({"encounter_id": [1, 2, 3, 4], "clinical_note_v2": notes}).to_parquet(
        tmp_path / "notes_v2_train.parquet", index=False
    )
    cfg = jina_client.JinaConfig("k", "jina-test", "r")
    post, calls = fake_post_factory()
    args = module.parse_args(
        [
            "--variant",
            "v2",
            "--data-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "2",
            "--workers",
            "2",
        ]
    )
    args.out.mkdir()
    conn = module.open_cache(args.out / module.CACHE_NAME)
    spent = [0]
    real_fill = module.fill_cache
    monkeypatch.setattr(
        module,
        "fill_cache",
        lambda *a, **k: real_fill(*a, **{**k, "post": post}) if k else real_fill(*a, post=post),
    )
    module.embed_split(args, cfg, conn, "train", spent)
    assert calls["texts"] == 3  # the duplicate note is embedded once
    vecs = np.load(args.out / "embeddings_jina_v5_small_v2_train.npy")
    assert vecs.shape == (4, 8) and np.allclose(vecs[0], vecs[2])
    # valid file + manifest: the second run must not call the API at all
    module.embed_split(args, cfg, conn, "train", spent)
    assert calls["texts"] == 3
    conn.close()
    nr.load_embeddings(args.out, "jina_v5_small", "v2", "train", np.array([1, 2, 3, 4]))


# --------------------------------------------------------------------------------------
# Notes v2 on the real splits (skipped when the parquets are absent, e.g. in CI)
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
REAL_SPLITS = [ROOT_DIR / "data" / "processed" / f"{s}.parquet" for s in ("train", "val", "test")]


@pytest.mark.skipif(
    not all(p.exists() for p in REAL_SPLITS), reason="real split parquets not present"
)
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_notes_v2_on_every_row_of_the_real_split(split):
    from src.note_rag_config import expected_split_rows, split_parquet

    df = pd.read_parquet(split_parquet(split))
    assert len(df) == expected_split_rows()[split]
    notes = notes_v2.build_notes(df)
    report = notes_v2.validate_notes(df, notes)
    notes_v2.assert_valid(report, split)
    assert report["rows"] == len(df)
