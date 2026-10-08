import json
from pathlib import Path
import sys
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_RAG_PARQUET,
    VAL_RAG_PARQUET,
)
from src.evaluate import evaluate_predictions


class ClinicalNotesDataset(Dataset):
    """PyTorch Dataset for clinical discharge notes tokenization."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def train_distilbert_classifier(
    train_parquet: Path = TRAIN_RAG_PARQUET,
    val_parquet: Path = VAL_RAG_PARQUET,
    model_name: str = "distilbert-base-uncased",
    sample_size_cpu: int = 300,
    val_sample_size_cpu: int = 150,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    device: Optional[str] = None,
) -> Dict:
    """Fine-tune DistilBERT sequence classifier on clinical discharge notes."""
    output_dir = MODELS_DIR / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DistilBERT Training Device: {device.upper()}", flush=True)

    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)

    if device == "cpu":
        print("WARNING: Running DistilBERT on CPU without GPU acceleration.", flush=True)
        print("CPU runs are restricted to smoke testing only. For full benchmark results, run notebooks/colab_transformer_training.ipynb on Colab GPU.", flush=True)
        if sample_size_cpu and len(train_df) > sample_size_cpu:
            print(f"CPU smoke-test mode: capping to {sample_size_cpu} train notes...", flush=True)
        # Stratified sample
        pos_train = train_df[train_df["readmitted_30d"] == 1]
        neg_train = train_df[train_df["readmitted_30d"] == 0]
        pos_n = int(sample_size_cpu * 0.114)
        neg_n = sample_size_cpu - pos_n
        train_df = pd.concat([pos_train.sample(n=min(len(pos_train), pos_n), random_state=RANDOM_SEED),
                              neg_train.sample(n=min(len(neg_train), neg_n), random_state=RANDOM_SEED)]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        pos_val = val_df[val_df["readmitted_30d"] == 1]
        neg_val = val_df[val_df["readmitted_30d"] == 0]
        pos_vn = int(val_sample_size_cpu * 0.114)
        neg_vn = val_sample_size_cpu - pos_vn
        val_df = pd.concat([pos_val.sample(n=min(len(pos_val), pos_vn), random_state=RANDOM_SEED),
                            neg_val.sample(n=min(len(neg_val), neg_vn), random_state=RANDOM_SEED)]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Fine-tuning on: Train={len(train_df)} notes, Val={len(val_df)} notes", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_dataset = ClinicalNotesDataset(train_df["clinical_note"], train_df["readmitted_30d"], tokenizer)
    val_dataset = ClinicalNotesDataset(val_df["clinical_note"], val_df["readmitted_30d"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class-weighted loss to handle imbalanced ~11% positive class
    weights = torch.tensor([1.0, 7.7], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Starting fine-tuning for {epochs} epochs...", flush=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Avg Train Loss: {total_loss/len(train_loader):.4f}", flush=True)

    # Evaluate on Validation notes
    model.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    metrics = evaluate_predictions(val_targets, val_probs)
    metrics.update({
        "model": "DistilBERT (Clinical Notes)",
        "rag_feature": "N/A (Text Model)",
    })

    print(f"Validation AUC: {metrics['auc_roc']} (CI: {metrics['auc_ci_95']}) | F1: {metrics['f1']} | Prec@10%: {metrics['precision_top_10pct']} (Lift: {metrics['lift_top_10pct']}x)")

    # Save fine-tuned weights and tokenizer
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved fine-tuned transformer weights to {output_dir}")

    # Append to ablation table
    ablation_file = METRICS_DIR / "ablation_table.json"
    if ablation_file.exists():
        with open(ablation_file, "r") as f:
            records = json.load(f)
    else:
        records = []

    records.append(metrics)
    with open(ablation_file, "w") as f:
        json.dump(records, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_distilbert_classifier()
