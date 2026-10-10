"""Embed clinical notes with a sentence-transformers model (self-contained, GPU friendly).

This script does not import anything from the repo, so it can run on a rented GPU box
or Colab with only the parquet files, this file and requirements-note-rag.txt.

Outputs (in --out):
    embeddings_{model_key}_{variant}_{split}.npy   float32, L2-normalized, parquet row order
    embeddings_{model_key}_{variant}_index.parquet columns encounter_id, split, row
    MANIFEST.json                                  rows, dim, sha256, seconds, model, device, versions

A split is skipped when its .npy file exists and matches the checksum recorded in MANIFEST.json.

Examples:
    python scripts/embed_notes.py --model minilm --variant v2 --data-dir data/processed --out out
    python scripts/embed_notes.py --model bge_small --variant v1 --limit 300 --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MODELS: Dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}
VARIANTS: Dict[str, Dict[str, str]] = {
    "v1": {"file": "{split}.parquet", "column": "clinical_note"},
    "v2": {"file": "notes_v2_{split}.parquet", "column": "clinical_note_v2"},
}
ID_COL = "encounter_id"
MANIFEST_NAME = "MANIFEST.json"
CHUNK_ROWS = 10_000


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_file(model_key: str, variant: str, split: str) -> str:
    return f"embeddings_{model_key}_{variant}_{split}.npy"


def find_input(data_dir: Path, name: str) -> Path:
    """Look for the parquet directly in data_dir, then in data_dir/note_rag."""
    for candidate in (data_dir / name, data_dir / "note_rag" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {name} under {data_dir} or {data_dir / 'note_rag'}")


def read_manifest(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"entries": {}}


def write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    tmp = out_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_dir / MANIFEST_NAME)


def is_split_done(out_dir: Path, file_name: str, expected_rows: int, model_name: str) -> bool:
    """True when the .npy exists and matches the manifest checksum, rows and model."""
    path = out_dir / file_name
    entry = read_manifest(out_dir)["entries"].get(file_name)
    if not path.exists() or entry is None:
        return False
    return (
        entry.get("rows") == expected_rows
        and entry.get("model_name") == model_name
        and entry.get("sha256") == sha256_of(path)
    )


def pick_batch_size(requested: str, device: str) -> int:
    if requested != "auto":
        return int(requested)
    if device != "cuda":
        return 64
    import torch

    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    return 512 if gib >= 12 else 256 if gib >= 6 else 128


def environment_versions() -> Dict[str, str]:
    import sentence_transformers
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }


def warn_if_cpu(device: str) -> None:
    if device == "cpu":
        bar = "!" * 72
        log(bar)
        log("WARNING: no CUDA GPU found. Embedding ~100k notes on CPU is very slow.")
        log("Use --limit N only for a smoke test, or run this on a GPU machine.")
        log(bar)


def encode_in_chunks(model: Any, notes: List[str], batch_size: int, label: str) -> np.ndarray:
    """Encode in chunks so progress can be logged; vectors are normalized float32."""
    parts: List[np.ndarray] = []
    start = time.time()
    for lo in range(0, len(notes), CHUNK_ROWS):
        chunk = notes[lo : lo + CHUNK_ROWS]
        vecs = model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        parts.append(vecs.astype(np.float32))
        done = lo + len(chunk)
        rate = done / max(time.time() - start, 1e-9)
        eta = (len(notes) - done) / rate
        log(f"[{label}] {done}/{len(notes)} rows | {rate:.0f} rows/s | eta {eta:.0f}s")
    return np.concatenate(parts, axis=0)


def update_index(out_dir: Path, model_key: str, variant: str, split: str, ids: np.ndarray) -> None:
    """Rewrite the index parquet, replacing the rows of this split."""
    path = out_dir / f"embeddings_{model_key}_{variant}_index.parquet"
    new = pd.DataFrame({ID_COL: ids, "split": split, "row": np.arange(len(ids), dtype=np.int64)})
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[old["split"] != split], new], ignore_index=True)
    new.to_parquet(path, index=False)


def embed_split(
    model: Any,
    args: argparse.Namespace,
    split: str,
    device: str,
    batch_size: int,
    versions: Dict[str, str],
) -> None:
    spec = VARIANTS[args.variant]
    model_name = MODELS[args.model]
    path = find_input(args.data_dir, spec["file"].format(split=split))
    df = pd.read_parquet(path, columns=[ID_COL, spec["column"]])
    if args.limit:
        df = df.head(args.limit)
    file_name = embedding_file(args.model, args.variant, split)
    if is_split_done(args.out, file_name, len(df), model_name):
        log(f"[{split}] {file_name} already valid, skipping")
        return

    log(f"[{split}] embedding {len(df)} notes from {path.name} with {model_name}")
    t0 = time.time()
    vecs = encode_in_chunks(model, df[spec["column"]].tolist(), batch_size, split)
    seconds = time.time() - t0
    norms = np.linalg.norm(vecs, axis=1)
    if not np.isfinite(vecs).all() or not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"[{split}] embeddings are not finite and unit norm")

    out_path = args.out / file_name
    np.save(out_path, vecs)
    update_index(args.out, args.model, args.variant, split, df[ID_COL].to_numpy())
    manifest = read_manifest(args.out)
    manifest["entries"][file_name] = {
        "rows": int(vecs.shape[0]),
        "dim": int(vecs.shape[1]),
        "sha256": sha256_of(out_path),
        "seconds": round(seconds, 1),
        "model_name": model_name,
        "model_key": args.model,
        "variant": args.variant,
        "split": split,
        "device": device,
        "batch_size": batch_size,
        "limit": args.limit,
        "versions": versions,
    }
    write_manifest(args.out, manifest)
    log(f"[{split}] saved {file_name} {vecs.shape} in {seconds:.0f}s")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/note_rag"))
    parser.add_argument("--limit", type=int, default=0, help="first N rows per split (smoke test)")
    parser.add_argument("--batch-size", default="auto", help="integer or 'auto'")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warn_if_cpu(device)
    args.out.mkdir(parents=True, exist_ok=True)
    batch_size = pick_batch_size(args.batch_size, device)
    log(f"model={MODELS[args.model]} variant={args.variant} device={device} batch={batch_size}")
    model = SentenceTransformer(MODELS[args.model], device=device)
    versions = environment_versions()
    for split in args.splits:
        embed_split(model, args, split, device, batch_size, versions)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
