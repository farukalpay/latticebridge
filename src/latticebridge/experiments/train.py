from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from latticebridge.data.records import read_jsonl
from latticebridge.models.prefix_lm import PrefixLMConfig, PrefixLanguageModel
from latticebridge.models.tokenizer import LatticeTokenizer


class PrefixDataset(Dataset):
    def __init__(self, records, tokenizer: LatticeTokenizer, max_tokens: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.bos_id = tokenizer.id_for("<bos>")
        self.src_id = tokenizer.id_for("<src>")
        self.tgt_id = tokenizer.id_for("<tgt>")
        self.eos_id = tokenizer.id_for("<eos>")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        record = self.records[index]
        prefix_ids = [self.bos_id, self.src_id] + self.tokenizer.encode(record.source_text) + [self.tgt_id]
        target_ids = self.tokenizer.encode(record.target_text) + [self.eos_id]
        full = (prefix_ids + target_ids)[: self.max_tokens]
        effective_prefix = min(len(prefix_ids), len(full) - 1)
        return {"input_ids": full, "prefix_cutoff": max(0, effective_prefix)}


def _collate(batch):
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len - 1), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)
    for row_idx, row in enumerate(batch):
        tokens = row["input_ids"]
        if len(tokens) < 2:
            continue
        input_slice = torch.tensor(tokens[:-1], dtype=torch.long)
        label_slice = torch.tensor(tokens[1:], dtype=torch.long)
        input_ids[row_idx, : len(input_slice)] = input_slice
        labels[row_idx, : len(label_slice)] = label_slice
        prefix_cutoff = int(row["prefix_cutoff"])
        labels[row_idx, : max(0, prefix_cutoff - 1)] = -100
    return input_ids, labels


def _run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total = 0.0
    count = 0
    for input_ids, labels in tqdm(loader, leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(1, count)


@torch.no_grad()
def _evaluate(model, loader, device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total = 0.0
    count = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, _ = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        total += float(loss.item())
        count += 1
    return total / max(1, count)


def train_model(
    *,
    processed_root: Path,
    tokenizer_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
    vocab_size: int = 4096,
    max_tokens: int = 160,
    batch_size: int = 32,
    epochs: int = 6,
    learning_rate: float = 3e-4,
) -> dict[str, object]:
    tokenizer = LatticeTokenizer.load(tokenizer_path)
    train_records = read_jsonl(processed_root / "train.jsonl")
    val_records = read_jsonl(processed_root / "validation.jsonl")

    train_dataset = PrefixDataset(train_records, tokenizer, max_tokens=max_tokens)
    val_dataset = PrefixDataset(val_records, tokenizer, max_tokens=max_tokens)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)

    config = PrefixLMConfig(vocab_size=tokenizer.vocab_size, d_model=256, hidden_size=384, num_layers=2, dropout=0.15)
    model = PrefixLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val = float("inf")
    history = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "history": history,
                },
                checkpoint_dir / "best.pt",
            )

    report = {
        "best_val_loss": best_val,
        "history": history,
        "checkpoint": str(checkpoint_dir / "best.pt"),
    }
    (checkpoint_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_trained_model(checkpoint_path: Path, device: torch.device) -> PrefixLanguageModel:
    payload = torch.load(checkpoint_path, map_location=device)
    config = PrefixLMConfig(**payload["config"])
    model = PrefixLanguageModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
