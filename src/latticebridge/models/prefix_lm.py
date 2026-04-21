from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PrefixLMConfig:
    vocab_size: int
    d_model: int = 256
    hidden_size: int = 384
    num_layers: int = 2
    dropout: float = 0.15


class PrefixLanguageModel(nn.Module):
    def __init__(self, config: PrefixLMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids)
        outputs, hidden = self.gru(embeddings, hidden)
        logits = self.lm_head(self.dropout(outputs))
        return logits, hidden

    def warm_start(self, prefix_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden = self(prefix_ids)
        return logits[:, -1, :], hidden

    def step(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden = self(token_ids.unsqueeze(1), hidden)
        return logits[:, -1, :], hidden
