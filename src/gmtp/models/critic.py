import torch
import torch.nn as nn
from RLAlg.nn.layers import CriticHead
from RLAlg.nn.steps import ValueStep
from RLAlg.normalizer import Normalizer

from .layers import MLPLayer, NormPosition

CRITIC_HIDDEN_DIM = 512


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()

        self.normlizer = Normalizer((obs_dim,))
        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
            MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
            MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
            MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        )
        self.head = CriticHead(CRITIC_HIDDEN_DIM)

    def forward(self, obs: torch.Tensor, update_normlizer: bool = False) -> ValueStep:
        obs = self.normlizer(obs, update=update_normlizer)
        x = self.encoder(obs)
        return self.head(x)
