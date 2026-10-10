"""Minimal Jina AI client (embeddings and reranking) using only the standard library.

The API key is read from the environment or a git-ignored .env file and is never logged or
included in error messages.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import BASE_DIR

API_BASE = "https://api.jina.ai/v1"
DEFAULT_TASK = "text-matching"  # symmetric note-to-note similarity
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
ENV_PATH = BASE_DIR / ".env"


class JinaError(RuntimeError):
    """Raised for API failures; never contains the API key."""


@dataclass(frozen=True)
class JinaConfig:
    api_key: str
    model: str
    rerank_model: str
    task: str = DEFAULT_TASK

    def __repr__(self) -> str:  # keep the key out of logs and tracebacks
        return f"JinaConfig(model={self.model!r}, rerank_model={self.rerank_model!r})"


Post = Callable[[str, Dict[str, Any], str], Dict[str, Any]]


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE lines into os.environ without overriding values already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def config_from_env(path: Path = ENV_PATH) -> JinaConfig:
    load_env_file(path)
    key = os.getenv("JINA_API_KEY")
    if not key:
        raise JinaError("JINA_API_KEY is not set (put it in the git-ignored .env file)")
    return JinaConfig(
        api_key=key,
        model=os.getenv("JINA_MODEL", "jina-embeddings-v5-text-small"),
        rerank_model=os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"),
    )


def post_json(
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    retries: int = 6,
    timeout: float = 120.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """POST JSON with exponential backoff on rate limits and server errors."""
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with opener(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_STATUS and attempt < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                sleep(float(retry_after) if retry_after else min(2.0**attempt, 60.0))
                continue
            raise JinaError(f"HTTP {exc.code} from {url.rsplit('/', 1)[-1]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < retries:
                sleep(min(2.0**attempt, 60.0))
                continue
            raise JinaError(f"network error contacting {url.rsplit('/', 1)[-1]}: {exc}") from None
    raise JinaError("unreachable")


def _default_post(url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    return post_json(url, payload, api_key)


def embed_texts(
    texts: Sequence[str], cfg: JinaConfig, post: Post = _default_post
) -> Tuple[np.ndarray, int]:
    """Embed texts; returns unit-norm float32 vectors in input order and the tokens billed."""
    payload = {
        "model": cfg.model,
        "task": cfg.task,
        "normalized": True,
        "embedding_type": "float",
        "input": list(texts),
    }
    result = post(f"{API_BASE}/embeddings", payload, cfg.api_key)
    data = sorted(result["data"], key=lambda item: item["index"])
    if len(data) != len(texts):
        raise JinaError(f"asked for {len(texts)} embeddings, got {len(data)}")
    vecs = np.asarray([item["embedding"] for item in data], dtype=np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    tokens = int(result.get("usage", {}).get("total_tokens", 0))
    return vecs.astype(np.float32), tokens


def rerank(
    query: str,
    documents: Sequence[str],
    cfg: JinaConfig,
    top_n: Optional[int] = None,
    post: Post = _default_post,
) -> List[Tuple[int, float]]:
    """Rerank documents for a query; returns (document index, relevance score), best first."""
    payload: Dict[str, Any] = {
        "model": cfg.rerank_model,
        "query": query,
        "documents": list(documents),
    }
    if top_n:
        payload["top_n"] = top_n
    result = post(f"{API_BASE}/rerank", payload, cfg.api_key)
    return [(int(r["index"]), float(r["relevance_score"])) for r in result["results"]]
