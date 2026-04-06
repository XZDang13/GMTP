from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .config import MotionVAEPretrainConfig
from .model import ReferenceMotionVAE
from .schema import MotionFeatureSchema

MOTION_VAE_CHECKPOINT_VERSION = 1
MOTION_ENCODER_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class MotionVAECheckpointV1:
    meta: dict[str, Any]
    model: dict[str, Any]
    schema: MotionFeatureSchema
    training: dict[str, Any]
    optimizer: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoint_version: int = MOTION_VAE_CHECKPOINT_VERSION
    checkpoint_type: str = "motion_vae"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "checkpoint_type": self.checkpoint_type,
            "meta": self.meta,
            "model": self.model,
            "schema": self.schema.to_dict(),
            "training": self.training,
            "optimizer": self.optimizer,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionVAECheckpointV1":
        if payload.get("checkpoint_version") != MOTION_VAE_CHECKPOINT_VERSION:
            raise ValueError("Unsupported motion VAE checkpoint version.")
        if payload.get("checkpoint_type") != "motion_vae":
            raise ValueError("Expected checkpoint_type='motion_vae'.")
        return cls(
            meta=dict(payload["meta"]),
            model=dict(payload["model"]),
            schema=MotionFeatureSchema.from_dict(payload["schema"]),
            training=dict(payload["training"]),
            optimizer=dict(payload.get("optimizer", {})),
            artifacts=dict(payload.get("artifacts", {})),
        )


@dataclass(frozen=True)
class MotionEncoderCheckpointV1:
    meta: dict[str, Any]
    model: dict[str, Any]
    schema: MotionFeatureSchema
    training: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoint_version: int = MOTION_ENCODER_CHECKPOINT_VERSION
    checkpoint_type: str = "motion_encoder"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "checkpoint_type": self.checkpoint_type,
            "meta": self.meta,
            "model": self.model,
            "schema": self.schema.to_dict(),
            "training": self.training,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionEncoderCheckpointV1":
        if payload.get("checkpoint_version") != MOTION_ENCODER_CHECKPOINT_VERSION:
            raise ValueError("Unsupported motion encoder checkpoint version.")
        if payload.get("checkpoint_type") != "motion_encoder":
            raise ValueError("Expected checkpoint_type='motion_encoder'.")
        return cls(
            meta=dict(payload["meta"]),
            model=dict(payload["model"]),
            schema=MotionFeatureSchema.from_dict(payload["schema"]),
            training=dict(payload.get("training", {})),
            artifacts=dict(payload.get("artifacts", {})),
        )


def _encoder_kwargs_from_config(config: MotionVAEPretrainConfig, *, input_dim: int) -> dict[str, Any]:
    return {
        "input_dim": int(input_dim),
        "window_length": int(config.data.past_frames),
        "latent_dim": int(config.model.latent_dim),
        "channels": tuple(int(value) for value in config.model.encoder_channels),
        "kernel_size": int(config.model.kernel_size),
        "stride": int(config.model.stride),
        "activation": str(config.model.activation),
    }


def _decoder_kwargs_from_config(config: MotionVAEPretrainConfig, *, target_dim: int) -> dict[str, Any]:
    return {
        "latent_dim": int(config.model.latent_dim),
        "future_frames": int(config.data.future_frames),
        "target_dim": int(target_dim),
        "hidden_dims": tuple(int(value) for value in config.model.decoder_hidden_dims),
        "activation": str(config.model.activation),
    }


def build_motion_vae_checkpoint(
    *,
    model: ReferenceMotionVAE,
    optimizer: torch.optim.Optimizer | None,
    schema: MotionFeatureSchema,
    config: MotionVAEPretrainConfig,
    epoch: int,
    best_metric: float,
    artifacts: dict[str, Any] | None = None,
) -> MotionVAECheckpointV1:
    return MotionVAECheckpointV1(
        meta={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "latent_dim": int(model.latent_dim),
            "past_frames": int(model.past_frames),
            "future_frames": int(model.future_frames),
            "encoder_kwargs": _encoder_kwargs_from_config(config, input_dim=schema.d_ref),
            "decoder_kwargs": _decoder_kwargs_from_config(config, target_dim=schema.d_target),
        },
        model={
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
        },
        schema=schema,
        training={
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "config": config.to_dict(),
        },
        optimizer=optimizer.state_dict() if optimizer is not None else {},
        artifacts=artifacts or {},
    )


def build_motion_encoder_checkpoint(
    *,
    model: ReferenceMotionVAE,
    schema: MotionFeatureSchema,
    config: MotionVAEPretrainConfig,
    epoch: int,
    best_metric: float,
    artifacts: dict[str, Any] | None = None,
) -> MotionEncoderCheckpointV1:
    return MotionEncoderCheckpointV1(
        meta={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "latent_dim": int(model.latent_dim),
            "encoder_kwargs": _encoder_kwargs_from_config(config, input_dim=schema.d_ref),
            "frozen": True,
        },
        model={
            "encoder": model.encoder.state_dict(),
        },
        schema=schema,
        training={
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "config": config.to_dict(),
        },
        artifacts=artifacts or {},
    )


def save_motion_vae_checkpoint(checkpoint: MotionVAECheckpointV1, path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_dict(), checkpoint_path)
    return checkpoint_path


def load_motion_vae_checkpoint(path: str | Path) -> MotionVAECheckpointV1:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Motion VAE checkpoint payload must be a dictionary.")
    return MotionVAECheckpointV1.from_dict(payload)


def save_motion_encoder_checkpoint(checkpoint: MotionEncoderCheckpointV1, path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_dict(), checkpoint_path)
    return checkpoint_path


def load_motion_encoder_checkpoint_v1(path: str | Path) -> MotionEncoderCheckpointV1:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Motion encoder checkpoint payload must be a dictionary.")
    return MotionEncoderCheckpointV1.from_dict(payload)
