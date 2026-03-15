import torch
import torch.nn as nn

from RLAlg.normalizer import Normalizer
from RLAlg.nn.layers import MLPLayer, GaussianHead, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep

class Actor(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super().__init__()

        self.normlizer = Normalizer((obs_dim,))

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
        )

        self.head = GaussianHead(512, action_dim)

    def forward(self, obs:torch.Tensor, action:torch.Tensor|None=None, update_normlizer:bool=False) -> StochasticContinuousPolicyStep:
        obs = self.normlizer(obs, update=update_normlizer)
        x = self.encoder(obs)
        step = self.head(x, action)

        return step
