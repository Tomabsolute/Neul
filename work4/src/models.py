from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CharLSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        logits = self.fc(self.dropout(output))
        return logits, hidden


class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center: torch.Tensor, pos_context: torch.Tensor, neg_context: torch.Tensor) -> torch.Tensor:
        center_emb = self.in_embed(center)
        pos_emb = self.out_embed(pos_context)
        neg_emb = self.out_embed(neg_context)

        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

