import torch
import torch.nn as nn
from RLAlg.utils import weight_init


class AdaIN(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()

        self.style = nn.Linear(cond_dim, feature_dim * 2)
        weight_init(self.style)

    def forward(self, x, cond):
        style = self.style(cond)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = 1 + gamma
        return gamma * x + beta


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()

        self.affine = nn.Linear(cond_dim, feature_dim * 2)
        weight_init(self.affine)

    def forward(self, x, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class AdaINBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()

        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.adain = AdaIN(dim, cond_dim)
        self.act = nn.SiLU()

        weight_init(self.fc)

    def forward(self, x, cond):
        h = self.fc(x)
        h = self.norm(h)
        h = self.adain(h, cond)
        h = self.act(h)
        return h


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int | None = None):
        super().__init__()

        hidden_dim = hidden_dim or dim * 2

        self.norm = nn.LayerNorm(dim)
        self.modulation = FiLM(dim, cond_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()
        self.res_scale = nn.Parameter(torch.full((dim,), 1e-2))

        weight_init(self.fc1)
        weight_init(self.fc2)

    def forward(self, x, cond):
        h = self.norm(x)
        h = self.modulation(h, cond)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return h


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


class AttnRes(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.key_norm = RMSNorm(dim)

    def forward(self, sources: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
        if not sources:
            raise ValueError("AttnRes requires at least one source tensor.")

        values = torch.stack(sources, dim=0)
        keys = self.key_norm(values)
        logits = torch.einsum("...d,n...d->n...", query, keys)
        attention = logits.softmax(dim=0)
        return torch.einsum("n...,n...d->...d", attention, values)


class BlockAttnRes(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.attn_res = AttnRes(dim)

    @staticmethod
    def build_sources(
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        sources = list(completed_blocks)
        if partial_block is not None:
            sources.append(partial_block)
        return sources

    def forward(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if not completed_blocks:
            raise ValueError("BlockAttnRes requires the initial embedding in completed_blocks.")
        return self.attn_res(self.build_sources(completed_blocks, partial_block), query)


class BlockAttnResFiLMStack(nn.Module):
    DEFAULT_BLOCK_SIZE = 4

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_layers: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}.")

        self.num_layers = num_layers
        self.block_size = block_size
        self.blocks = nn.ModuleList(
            [FiLMResBlock(dim, cond_dim, hidden_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.query_projs = nn.ModuleList([nn.Linear(cond_dim, dim) for _ in range(num_layers)])
        self.attn_res = BlockAttnRes(dim)

        for query_proj in self.query_projs:
            nn.init.zeros_(query_proj.weight)
            nn.init.zeros_(query_proj.bias)

    def _is_block_boundary(self, layer_index: int) -> bool:
        return (layer_index + 1) % self.block_size == 0 or layer_index + 1 == self.num_layers

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        completed_blocks = [x]
        partial_block = None

        for layer_index, block in enumerate(self.blocks):
            query = self.query_projs[layer_index](cond)
            h = self.attn_res(completed_blocks, partial_block, query)
            dx = block(h, cond)
            x = h + block.res_scale * dx
            partial_block = x if partial_block is None else partial_block + x

            if self._is_block_boundary(layer_index):
                completed_blocks.append(partial_block)
                partial_block = None

        return x
