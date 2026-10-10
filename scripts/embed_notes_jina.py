"""Embed clinical notes with the Jina API and store the vectors in a SQLite cache.

No GPU is needed. Vectors are saved with the same naming convention as scripts/embed_notes.py
(model_key = jina_v5_small), so the rest of the pipeline treats them like any other retriever:

    embeddings_jina_v5_small_{variant}_{split}.npy
    embeddings_jina_v5_small_{variant}_index.parquet
    MANIFEST.json
    embedding_cache_jina_v5_small.sqlite      text-hash -> vector (resume + de-duplication)

Identical notes are embedded once, which cuts the number of billed tokens. The API key is read
from JINA_API_KEY (environment or git-ignored .env) and is never printed or saved.

    python scripts/embed_notes_jina.py --variant v2 --limit 300 --out /tmp/jina_smoke
    python scripts/embed_notes_jina.py --variant v2 --token-budget 15000000
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embed_notes import (  # noqa: E402  (sibling script)
    ID_COL,
    VARIANTS,
    embedding_file,
    find_input,
    is_split_done,
    log,
    read_manifest,
    sha256_of,
    update_index,
    write_manifest,
)

from src import jina_client  # noqa: E402
from src.note_rag_config import NOTE_RAG_DATA_DIR  # noqa: E402

MODEL_KEY = "jina_v5_small"
CACHE_NAME = f"embedding_cache_{MODEL_KEY}.sqlite"


class TokenBudgetExceeded(RuntimeError):
    """The billed tokens passed --token-budget; progress so far is kept in the cache."""


def text_key(model: str, task: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{task}|{text}".encode("utf-8")).hexdigest()


def open_cache(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    return conn


def cached_keys(conn: sqlite3.Connection, keys: Sequence[str], step: int = 500) -> set:
    found = set()
    for lo in range(0, len(keys), step):
        part = list(keys[lo : lo + step])
        marks = ",".join("?" * len(part))
        found.update(
            r[0] for r in conn.execute(f"SELECT key FROM cache WHERE key IN ({marks})", part)
        )
    return found


def fetch_vectors(
    conn: sqlite3.Connection, keys: Sequence[str], step: int = 500
) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for lo in range(0, len(keys), step):
        part = list(keys[lo : lo + step])
        marks = ",".join("?" * len(part))
        out.update(conn.execute(f"SELECT key, vec FROM cache WHERE key IN ({marks})", part))
    return out


def fill_cache(
    conn: sqlite3.Connection,
    texts_by_key: Dict[str, str],
    cfg: jina_client.JinaConfig,
    batch_size: int,
    workers: int,
    token_budget: Optional[int],
    spent: List[int],
    post=None,
) -> None:
    """Embed every text missing from the cache, a few requests in parallel."""
    already = cached_keys(conn, list(texts_by_key))
    missing = [k for k in texts_by_key if k not in already]
    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
    log(
        f"{len(texts_by_key)} unique texts, {len(texts_by_key) - len(missing)} cached, "
        f"{len(missing)} to embed in {len(batches)} requests"
    )
    call = (
        (lambda keys: jina_client.embed_texts([texts_by_key[k] for k in keys], cfg, post))
        if post
        else (lambda keys: jina_client.embed_texts([texts_by_key[k] for k in keys], cfg))
    )
    start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for lo in range(0, len(batches), workers):
            wave = batches[lo : lo + workers]
            for keys, (vecs, tokens) in zip(wave, pool.map(call, wave)):
                conn.executemany(
                    "INSERT OR REPLACE INTO cache(key, vec) VALUES (?, ?)",
                    [(k, v.tobytes()) for k, v in zip(keys, vecs)],
                )
                spent[0] += tokens
                done += len(keys)
            conn.commit()
            rate = done / max(time.time() - start, 1e-9)
            log(f"  embedded {done}/{len(missing)} | {rate:.0f} texts/s | tokens billed {spent[0]}")
            if token_budget is not None and spent[0] >= token_budget:
                raise TokenBudgetExceeded(f"billed {spent[0]} tokens, budget {token_budget}")


def embed_split(args, cfg, conn, split: str, spent: List[int]) -> None:
    spec = VARIANTS[args.variant]
    path = find_input(args.data_dir, spec["file"].format(split=split))
    df = pd.read_parquet(path, columns=[ID_COL, spec["column"]])
    if args.limit:
        df = df.head(args.limit)
    file_name = embedding_file(MODEL_KEY, args.variant, split)
    if is_split_done(args.out, file_name, len(df), cfg.model):
        log(f"[{split}] {file_name} already valid, skipping")
        return
    notes = df[spec["column"]].tolist()
    keys = [text_key(cfg.model, cfg.task, n) for n in notes]
    texts_by_key = dict(zip(keys, notes))
    t0 = time.time()
    tokens_before = spent[0]
    fill_cache(conn, texts_by_key, cfg, args.batch_size, args.workers, args.token_budget, spent)
    stored = fetch_vectors(conn, list(texts_by_key))
    vecs = np.stack([np.frombuffer(stored[k], dtype=np.float32) for k in keys])
    if not np.isfinite(vecs).all() or not np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"[{split}] embeddings are not finite and unit norm")
    out_path = args.out / file_name
    np.save(out_path, vecs)
    update_index(args.out, MODEL_KEY, args.variant, split, df[ID_COL].to_numpy())
    manifest = read_manifest(args.out)
    manifest["entries"][file_name] = {
        "rows": int(vecs.shape[0]),
        "dim": int(vecs.shape[1]),
        "sha256": sha256_of(out_path),
        "seconds": round(time.time() - t0, 1),
        "model_name": cfg.model,
        "model_key": MODEL_KEY,
        "variant": args.variant,
        "split": split,
        "device": "jina-api",
        "task": cfg.task,
        "unique_texts": len(texts_by_key),
        "api_tokens_billed_this_split": spent[0] - tokens_before,
        "limit": args.limit,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    write_manifest(args.out, manifest)
    log(f"[{split}] saved {file_name} {vecs.shape}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=NOTE_RAG_DATA_DIR)
    parser.add_argument("--limit", type=int, default=0, help="first N rows per split (smoke test)")
    parser.add_argument("--batch-size", type=int, default=64, help="texts per API request")
    parser.add_argument("--workers", type=int, default=4, help="parallel API requests")
    parser.add_argument(
        "--token-budget", type=int, default=None, help="stop after this many billed tokens"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = jina_client.config_from_env()
    args.out.mkdir(parents=True, exist_ok=True)
    conn = open_cache(args.out / CACHE_NAME)
    spent = [0]
    log(f"model={cfg.model} task={cfg.task} variant={args.variant} workers={args.workers}")
    try:
        for split in args.splits:
            embed_split(args, cfg, conn, split, spent)
    except TokenBudgetExceeded as exc:
        log(f"STOPPED: {exc}. Re-run to continue; cached vectors are kept.")
        return 2
    finally:
        conn.close()
    log(f"done; tokens billed in this run: {spent[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
