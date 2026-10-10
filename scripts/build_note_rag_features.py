"""Build leakage-safe note-embedding RAG features for train, val and test.

Reads validated embeddings, writes data/processed/note_rag/note_rag_features_{split}.parquet
(encounter_id plus feature columns, no label column in any file) and an assertions JSON.

    python scripts/build_note_rag_features.py            # uses reports/note_rag/selected_retriever.json
    python scripts/build_note_rag_features.py --model minilm --variant v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import note_rag as nr
from src.config import MODELS_DIR
from src.note_rag_config import (
    FEATURE_NAMES,
    ID_COL,
    K_GRID,
    LABEL_COL,
    NOTE_RAG_DATA_DIR,
    NOTE_RAG_REPORT_DIR,
    PRIMARY_FEATURE,
    SEED,
    features_path,
)
from src.note_rag_io import load_split_meta, meta_arrays

SELECTION_PATH = NOTE_RAG_REPORT_DIR / "selected_retriever.json"
ASSERTIONS_PATH = NOTE_RAG_REPORT_DIR / "note_rag_assertions.json"
FEATURES_META_PATH = NOTE_RAG_REPORT_DIR / "features_meta.json"
SQLITE_STORE_PATH = MODELS_DIR / "note_rag" / "note_rag_train.sqlite"
BRUTE_FORCE_ROWS = 200


def resolve_retriever(model: Optional[str], variant: Optional[str]) -> Dict[str, str]:
    if model and variant:
        return {"model_key": model, "variant": variant}
    if not SELECTION_PATH.exists():
        raise SystemExit(
            "No --model/--variant given and no selected_retriever.json; run selection."
        )
    chosen = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selected"]
    return {"model_key": chosen["model_key"], "variant": chosen["variant"]}


def save_features(split: str, ids: np.ndarray, frame: pd.DataFrame) -> Path:
    out = frame.copy()
    out.insert(0, ID_COL, ids)
    assert LABEL_COL not in out.columns
    path = features_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def sqlite_cross_check(train, emb_train, emb_val, val, val_idx, val_sims) -> Dict[str, Any]:
    """Recall of the SQLite (sqlite-vec) store against exact search; honest when unavailable."""
    SQLITE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = nr.build_sqlite_store(
            SQLITE_STORE_PATH, emb_train, train["ids"], train["patients"], train["y"]
        )
    except nr.SqliteStoreUnavailable as exc:
        return {"status": "unavailable", "reason": str(exc)}
    report = nr.sqlite_recall_at_k(
        conn, emb_train, emb_val, val["patients"], val_idx, val_sims, n_queries=500
    )
    conn.close()
    report["status"] = "ok"
    report["passed"] = bool(report["tie_aware_recall_at_k"] >= 0.99)
    return report


def run_checks(retriever, train, val, test, emb, tr, queries) -> Dict[str, Any]:
    """Run every leakage and sanity assertion; raises on the first hard failure."""
    k_max = max(K_GRID)
    checks: Dict[str, Any] = {"retriever": retriever, "k_grid": list(K_GRID), "seed": SEED}
    checks["same_patient_rows_checked"] = {
        "train": nr.assert_no_same_patient(tr.idx, train["patients"], train["patients"]),
        "val": nr.assert_no_same_patient(queries["val"][0], val["patients"], train["patients"]),
        "test": nr.assert_no_same_patient(queries["test"][0], test["patients"], train["patients"]),
    }
    nr.assert_folds_patient_disjoint(tr.folds, train["patients"], len(train["y"]))
    checks["oof_folds"] = {"n_folds": len(tr.folds), "patient_disjoint": True, "partition": True}
    nr.assert_neighbors_inside_index(tr.idx, tr.folds)
    nr.assert_same_k_grid({"train": K_GRID, "val": K_GRID, "test": K_GRID})
    checks["same_k_grid_all_splits"] = True
    idxs = {"train": tr.idx, "val": queries["val"][0], "test": queries["test"][0]}
    frames = {"train": tr.frame, "val": queries["val"][2], "test": queries["test"][2]}
    for split in frames:
        nr.assert_full_neighbors(idxs[split], k_max)
        nr.assert_features_valid(frames[split], n_neighbors_full=True)
    checks["no_nan_and_rates_on_k_grid"] = True
    checks["index_is_train_only"] = {
        "index_rows": int(len(emb["train"])),
        "train_rows": int(len(train["y"])),
        "val_and_test_queries_use_train_index": True,
    }
    checks["test_labels_read"] = False
    rows = np.random.default_rng(SEED).choice(len(train["y"]), BRUTE_FORCE_ROWS, replace=False)
    checks["brute_force"] = nr.check_oof_against_brute_force(
        emb["train"], train["patients"], train["y"], tr.folds, tr.idx, tr.frame, rows
    )
    control = nr.shuffled_label_control(
        queries["val"][0], queries["val"][1], train["y"], val["y"], seed=SEED
    )
    tolerance = nr.null_auc_tolerance(val["y"])
    checks["negative_control"] = {
        "val_auc_shuffled_train_labels": control,
        "tolerance_around_0_5": tolerance,
        "passed": bool(abs(control - 0.5) <= tolerance),
    }
    checks["univariate_auc_rate_k20"] = {
        "train_oof": nr.univariate_auc(train["y"], tr.frame[PRIMARY_FEATURE].to_numpy()),
        "val": nr.univariate_auc(val["y"], queries["val"][2][PRIMARY_FEATURE].to_numpy()),
    }
    checks["sqlite_vec_cross_check"] = sqlite_cross_check(
        train, emb["train"], emb["val"], val, queries["val"][0], queries["val"][1]
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--emb-dir", type=Path, default=NOTE_RAG_DATA_DIR)
    args = parser.parse_args()
    retriever = resolve_retriever(args.model, args.variant)
    key, var = retriever["model_key"], retriever["variant"]
    print(f"retriever: notes {var} with {key}", flush=True)

    metas = {s: load_split_meta(s) for s in ("train", "val", "test")}
    assert LABEL_COL not in metas["test"].columns, "test labels must not be loaded"
    train, val, test = (meta_arrays(metas[s]) for s in ("train", "val", "test"))
    emb = {
        s: nr.load_embeddings(args.emb_dir, key, var, s, m["ids"])
        for s, m in (("train", train), ("val", val), ("test", test))
    }
    print("embeddings validated for train, val, test", flush=True)

    t0 = time.time()
    tr = nr.build_train_features(emb["train"], train["patients"], train["y"])
    print(f"train OOF done in {time.time() - t0:.0f}s", flush=True)
    queries = {}
    for split, meta in (("val", val), ("test", test)):
        queries[split] = nr.build_query_features(
            emb[split], meta["patients"], emb["train"], train["patients"], train["y"]
        )
        print(f"{split} queried against the TRAIN index", flush=True)

    checks = run_checks(retriever, train, val, test, emb, tr, queries)
    failed = [
        name
        for name in ("negative_control", "sqlite_vec_cross_check")
        if checks[name].get("passed") is False
    ]
    checks["all_passed"] = not failed
    NOTE_RAG_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSERTIONS_PATH.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if failed:
        print(f"FAILED checks: {failed}; see {ASSERTIONS_PATH.name}")
        return 1

    frames = {"train": tr.frame, "val": queries["val"][2], "test": queries["test"][2]}
    for split, meta in (("train", train), ("val", val), ("test", test)):
        save_features(split, meta["ids"], frames[split])
    np.savez_compressed(
        args.emb_dir / "val_neighbors.npz",
        idx=queries["val"][0].astype(np.int32),
        sims=queries["val"][1].astype(np.float32),
    )
    FEATURES_META_PATH.write_text(
        json.dumps(
            {
                "retriever": retriever,
                "features": list(FEATURE_NAMES),
                "k_grid": list(K_GRID),
                "rows": {s: int(len(f)) for s, f in frames.items()},
                "note": "Feature files carry encounter_id and features only; no label column.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"All checks passed. Assertions: {ASSERTIONS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
