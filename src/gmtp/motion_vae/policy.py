from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .checkpoints import load_motion_encoder_checkpoint_v1
from .model import TemporalConvEncoder
from .schema import MotionFeatureSchema


def freeze_motion_encoder(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class FrozenMotionEncoder(nn.Module):
    def __init__(self, encoder: TemporalConvEncoder, schema: MotionFeatureSchema) -> None:
        super().__init__()
        self.encoder = freeze_motion_encoder(encoder)
        self.schema = schema
        self.latent_dim = int(self.encoder.latent_dim)
        self.register_buffer(
            "reference_mean",
            torch.as_tensor(schema.reference_mean, dtype=torch.float32).reshape(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "reference_std",
            torch.as_tensor(schema.reference_std, dtype=torch.float32).reshape(1, 1, -1),
            persistent=False,
        )

    def normalize_reference(self, reference: torch.Tensor) -> torch.Tensor:
        return (reference - self.reference_mean) / self.reference_std

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        reference = torch.as_tensor(reference, dtype=torch.float32, device=self.reference_mean.device)
        return self.encoder.encode(self.normalize_reference(reference), deterministic=True)


def _build_encoder_from_checkpoint(path: str | Path, *, device: torch.device) -> tuple[TemporalConvEncoder, MotionFeatureSchema]:
    checkpoint = load_motion_encoder_checkpoint_v1(path)
    encoder_kwargs = dict(checkpoint.meta["encoder_kwargs"])
    encoder = TemporalConvEncoder(
        input_dim=int(encoder_kwargs["input_dim"]),
        window_length=int(encoder_kwargs["window_length"]),
        latent_dim=int(encoder_kwargs["latent_dim"]),
        channels=tuple(int(value) for value in encoder_kwargs["channels"]),
        kernel_size=int(encoder_kwargs["kernel_size"]),
        stride=int(encoder_kwargs["stride"]),
        activation=str(encoder_kwargs["activation"]),
    ).to(device)
    encoder.load_state_dict(checkpoint.model["encoder"])
    encoder.eval()
    return encoder, checkpoint.schema


def build_frozen_motion_encoder(path: str | Path, device: torch.device | str) -> FrozenMotionEncoder:
    resolved_device = torch.device(device)
    encoder, schema = _build_encoder_from_checkpoint(path, device=resolved_device)
    return FrozenMotionEncoder(encoder, schema).to(resolved_device)
