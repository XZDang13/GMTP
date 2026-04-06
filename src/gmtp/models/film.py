import torch
import torch.nn as nn
from RLAlg.utils import weight_init


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()

        self.affine = nn.Linear(cond_dim, feature_dim * 2)
        weight_init(self.affine)

    def forward(self, x, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


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


class FiLMResStack(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_layers: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.num_layers = num_layers
        self.blocks = nn.ModuleList(
            [FiLMResBlock(dim, cond_dim, hidden_dim=hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            dx = block(x, cond)
            x = x + block.res_scale * dx
        return x
