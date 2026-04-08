from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from RLAlg.nn.layers import NormPosition
from RLAlg.utils import weight_init


class AmpSafeRMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and x.dtype != self.weight.dtype:
            normalized = super().forward(x.to(dtype=self.weight.dtype))
            return normalized.to(dtype=x.dtype)
        return super().forward(x)


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activate_func: Optional[nn.Module] = None,
        norm_position: NormPosition = NormPosition.NONE,
    ) -> None:
        super().__init__()
        if activate_func is None:
            activate_func = nn.Identity()
        if not isinstance(norm_position, NormPosition):
            raise TypeError(f"norm_position must be NormPosition, got {type(norm_position)}.")

        use_norm = norm_position is not NormPosition.NONE
        self.pre_norm = AmpSafeRMSNorm(in_dim) if use_norm and norm_position is NormPosition.PRE else nn.Identity()
        self.post_norm = (
            AmpSafeRMSNorm(out_dim) if use_norm and norm_position is NormPosition.POST else nn.Identity()
        )
        self.linear = nn.Linear(in_dim, out_dim, bias=not use_norm)
        self.activate_func = activate_func

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_init(self.linear)
        if not isinstance(self.pre_norm, nn.Identity):
            self.pre_norm.reset_parameters()
        if not isinstance(self.post_norm, nn.Identity):
            self.post_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.linear(x)
        x = self.post_norm(x)
        x = self.activate_func(x)
        return x


__all__ = ["AmpSafeRMSNorm", "MLPLayer", "NormPosition"]
