from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .checkpoints import load_motion_mae_encoder_checkpoint
from .model import ReferenceMotionMAE
from .schema import MotionFeatureSchema

MOTION_MAE_TOKEN_ENCODER_KEYS = (
    "input_proj.",
    "encoder_position_embedding",
    "encoder.",
)


def freeze_motion_encoder(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class FrozenMotionMAEEncoder(nn.Module):
    def __init__(self, encoder: ReferenceMotionMAE, schema: MotionFeatureSchema) -> None:
        super().__init__()
        self.encoder = freeze_motion_encoder(encoder)
        self.schema = schema
        self.token_dim = int(self.encoder.d_model)
        self.num_heads = int(self.encoder.nhead)
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
        return self.encoder.encode_visible(self.normalize_reference(reference))


def _is_token_encoder_key(key: str) -> bool:
    return key == "encoder_position_embedding" or key.startswith(MOTION_MAE_TOKEN_ENCODER_KEYS)


def _build_model_from_encoder_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[ReferenceMotionMAE, MotionFeatureSchema]:
    checkpoint = load_motion_mae_encoder_checkpoint(path)
    model_kwargs = dict(checkpoint.meta["model_kwargs"])
    encoder_state = checkpoint.model["encoder_state"]

    model = ReferenceMotionMAE(**model_kwargs)
    model_state = model.state_dict()
    required_encoder_keys = [key for key in model_state if _is_token_encoder_key(key)]
    missing_encoder_keys = [key for key in required_encoder_keys if key not in encoder_state]
    if missing_encoder_keys:
        raise ValueError(
            "Motion MAE encoder checkpoint is missing token-encoder weights. "
            "Regenerate or retrain the MAE encoder checkpoint. "
            f"Missing keys: {missing_encoder_keys}."
        )
    token_encoder_state = {
        key: value
        for key, value in encoder_state.items()
        if _is_token_encoder_key(key) and key in model_state
    }
    model.load_state_dict(token_encoder_state, strict=False)
    model.to(device)
    model.eval()
    return model, checkpoint.schema


def build_frozen_motion_mae_encoder(path: str | Path, device: torch.device | str) -> FrozenMotionMAEEncoder:
    resolved_device = torch.device(device)
    encoder, schema = _build_model_from_encoder_checkpoint(path, device=resolved_device)
    return FrozenMotionMAEEncoder(encoder, schema).to(resolved_device)
