from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoints import load_motion_mae_encoder_checkpoint
from .model import ReferenceMotionMAE
from .schema import MotionFeatureSchema


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
        return self.encoder.encode(self.normalize_reference(reference))


def _build_model_from_encoder_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[ReferenceMotionMAE, MotionFeatureSchema]:
    checkpoint = load_motion_mae_encoder_checkpoint(path)
    model_kwargs = dict(checkpoint.meta["model_kwargs"])
    model = ReferenceMotionMAE(**model_kwargs).to(device)
    model.load_state_dict(checkpoint.model["encoder_state"], strict=False)
    model.eval()
    return model, checkpoint.schema


def build_frozen_motion_mae_encoder(path: str | Path, device: torch.device | str) -> FrozenMotionMAEEncoder:
    resolved_device = torch.device(device)
    encoder, schema = _build_model_from_encoder_checkpoint(path, device=resolved_device)
    return FrozenMotionMAEEncoder(encoder, schema).to(resolved_device)


@torch.no_grad()
def export_motion_mae_latents(
    dataset: Any,
    encoder: FrozenMotionMAEEncoder,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    latents = []
    motion_names: list[str] = []
    motion_files: list[str] = []
    centers: list[int] = []
    for batch in loader:
        latent = encoder(batch["reference"].to(device=encoder.reference_mean.device, non_blocking=True))
        latents.append(latent.cpu())
        motion_names.extend([str(item) for item in batch["motion_name"]])
        motion_files.extend([str(item) for item in batch["motion_file"]])
        centers.extend(int(item) for item in batch["center_t"].tolist())

    return {
        "latents": torch.cat(latents, dim=0).numpy() if latents else np.zeros((0, encoder.latent_dim), dtype=np.float32),
        "motion_names": np.asarray(motion_names),
        "motion_files": np.asarray(motion_files),
        "center_t": np.asarray(centers, dtype=np.int64),
    }
