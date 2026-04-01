import torch
import torch.nn as nn
from RLAlg.nn.layers import CriticHead, MLPLayer, NormPosition
from RLAlg.nn.steps import ValueStep
from RLAlg.normalizer import Normalizer


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()

        self.normlizer = Normalizer((obs_dim,))
        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
        )
        self.head = CriticHead(512)

    def forward(self, obs: torch.Tensor, update_normlizer: bool = False) -> ValueStep:
        obs = self.normlizer(obs, update=update_normlizer)
        x = self.encoder(obs)
        return self.head(x)
