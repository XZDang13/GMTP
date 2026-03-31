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


class AdaINResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.adain = AdaIN(dim, cond_dim)
        self.act = nn.SiLU()

        weight_init(self.fc1)
        weight_init(self.fc2)

    def forward(self, x, cond):
        identity = x
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.adain(h, cond)
        return h + identity
