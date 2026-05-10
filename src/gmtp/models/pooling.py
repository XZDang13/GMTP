from __future__ import annotations

from enum import StrEnum

import torch
import torch.nn as nn


class EncoderPoolingType(StrEnum):
    LEARNED = "learned"
    LAST_TOKEN = "last_token"


def normalize_encoder_pooling_type(
    encoder_pooling_type: str | EncoderPoolingType | None,
) -> EncoderPoolingType:
    if encoder_pooling_type is None:
        return EncoderPoolingType.LEARNED

    normalized = str(encoder_pooling_type).lower().replace("-", "_")
    alias_map = {
        "learned": EncoderPoolingType.LEARNED,
        "attention": EncoderPoolingType.LEARNED,
        "attention_pool": EncoderPoolingType.LEARNED,
        "learned_query": EncoderPoolingType.LEARNED,
        "learned_query_attention": EncoderPoolingType.LEARNED,
        "last": EncoderPoolingType.LAST_TOKEN,
        "last_token": EncoderPoolingType.LAST_TOKEN,
        "final": EncoderPoolingType.LAST_TOKEN,
        "final_token": EncoderPoolingType.LAST_TOKEN,
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported encoder pooling type '{encoder_pooling_type}'. "
            "Expected one of: 'learned', 'last_token'."
        ) from exc


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


class LastTokenPool(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        embed_dim = int(embed_dim)
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        self.embed_dim = embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected last-token-pool tokens shape [B, T, D], got {tuple(tokens.shape)}.")
        if int(tokens.shape[-1]) != self.embed_dim:
            raise ValueError(f"Expected last-token-pool token dim {self.embed_dim}, got {tokens.shape[-1]}.")
        if int(tokens.shape[1]) < 1:
            raise ValueError("Expected last-token-pool token sequence length to be positive.")

        pooled = tokens[:, -1, :]
        if return_weights:
            weights = tokens.new_zeros((tokens.shape[0], tokens.shape[1]))
            weights[:, -1] = 1.0
            return pooled, weights
        return pooled
