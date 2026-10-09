"""CPU Smoke Test for Clinical Notes Transformer.

This module provides a CPU-only smoke test to verify tokenization, data loading,
and model forward/backward passes on a small subset of clinical notes (< 5,000 notes).

CRITICAL NOTE:
This is exclusively for pipeline smoke testing and CI sanity checks.
Full benchmark evaluation of the DistilBERT model occurs via Google Colab GPU
(notebooks/colab_transformer_training.ipynb) on the full validation split.
Results here are saved ONLY to reports/metrics/transformer_smoke_test.json
and NEVER into the ablation table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    METRICS_DIR,
    RANDOM_SEED,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)
from src.evaluate import evaluate_predictions

SMOKE_TEST_METRICS_PATH = METRICS_DIR / "transformer_smoke_test.json"


class ClinicalNotesDataset(Dataset):
    """PyTorch Dataset for clinical discharge notes tokenization."""

    def __init__(
        self,
        texts: list[str] | pd.Series,
        labels: list[int] | pd.Series | np.ndarray,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def sample_stratified_subset(
    df: pd.DataFrame,
    target_col: str,
    max_samples: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample a balanced/stratified subset up to max_samples for smoke testing."""
    if len(df) <= max_samples:
        return df.copy().reset_index(drop=True)

    pos_df = df[df[target_col] == 1]
    neg_df = df[df[target_col] == 0]
    pos_ratio = len(pos_df) / len(df)
    pos_n = int(max_samples * pos_ratio)
    neg_n = max_samples - pos_n

    sampled_pos = pos_df.sample(n=min(len(pos_df), pos_n), random_state=seed)
    sampled_neg = neg_df.sample(n=min(len(neg_df), neg_n), random_state=seed)
    combined = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=seed)
    return combined.reset_index(drop=True)


def run_cpu_smoke_test(
    train_parquet: Path = TRAIN_RAG_PARQUET,
    val_parquet: Path = VAL_RAG_PARQUET,
    model_name: str = "distilbert-base-uncased",
    train_samples: int = 200,
    val_samples: int = 100,
    batch_size: int = 16,
    max_length: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a CPU smoke test on a small subset of notes (< 5,000).

    Saves metrics ONLY to reports/metrics/transformer_smoke_test.json.
    Never alters the production ablation table.
    """
    assert train_samples < 5000, f"train_samples must be < 5000 for smoke test, got {train_samples}"
    assert val_samples < 5000, f"val_samples must be < 5000 for smoke test, got {val_samples}"

    print("=" * 70, flush=True)
    print(" LOUD WARNING: RUNNING TRANSFORMER CPU SMOKE TEST ONLY!", flush=True)
    print(" This is a sanity check on a tiny sample and NOT a production model.", flush=True)
    print(" Benchmark models are trained on Colab GPU on full datasets.", flush=True)
    print(f" Writing output strictly to: {SMOKE_TEST_METRICS_PATH}", flush=True)
    print("=" * 70, flush=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_parquet(train_parquet, columns=["clinical_note", "readmitted_30d"])
    val_df = pd.read_parquet(val_parquet, columns=["clinical_note", "readmitted_30d"])

    train_subset = sample_stratified_subset(train_df, "readmitted_30d", train_samples)
    val_subset = sample_stratified_subset(val_df, "readmitted_30d", val_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_ds = ClinicalNotesDataset(
        train_subset["clinical_note"],
        train_subset["readmitted_30d"],
        tokenizer,
        max_length=max_length,
    )
    val_ds = ClinicalNotesDataset(
        val_subset["clinical_note"],
        val_subset["readmitted_30d"],
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    # 1 mini-batch step to test forward + backward
    batch = next(iter(train_loader))
    optimizer.zero_grad()
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    loss = criterion(outputs.logits, batch["label"].to(device))
    loss.backward()
    optimizer.step()

    # Quick eval loop
    model.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for val_batch in val_loader:
            out = model(
                input_ids=val_batch["input_ids"].to(device),
                attention_mask=val_batch["attention_mask"].to(device),
            )
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_targets.extend(val_batch["label"].cpu().numpy())

    y_val = np.array(val_targets)
    y_prob = np.array(val_probs)

    smoke_metrics = evaluate_predictions(y_val, y_prob)
    smoke_results = {
        "mode": "CPU_SMOKE_TEST",
        "model_name": model_name,
        "train_samples_used": len(train_subset),
        "val_samples_used": len(val_subset),
        "smoke_test_loss": float(loss.item()),
        "auc_roc": smoke_metrics["auc_roc"],
        "auc_ci_95": smoke_metrics["auc_ci_95"],
        "note": "Smoke test sanity check only; not for clinical or ablation evaluation.",
    }

    SMOKE_TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(smoke_results, f, indent=2)

    print(f"Smoke test successful! Saved results to {SMOKE_TEST_METRICS_PATH}", flush=True)
    return smoke_results


if __name__ == "__main__":
    run_cpu_smoke_test()
