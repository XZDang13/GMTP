from __future__ import annotations

import torch
import torch.nn as nn


class LearnedQueryAttentionPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if num_heads < 1:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected attention-pool tokens shape [B, T, D], got {tuple(tokens.shape)}.")
        if int(tokens.shape[-1]) != self.embed_dim:
            raise ValueError(f"Expected attention-pool token dim {self.embed_dim}, got {tokens.shape[-1]}.")

        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, weights = self.attention(
            query,
            tokens,
            tokens,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        pooled = pooled.squeeze(1)
        if return_weights:
            return pooled, weights.squeeze(1)
        return pooled
