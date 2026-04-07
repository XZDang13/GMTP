from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .config import MotionMAEPretrainConfig
from .model import ReferenceMotionMAE
from .schema import MotionFeatureSchema

MOTION_MAE_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class MotionMAECheckpointV1:
    meta: dict[str, Any]
    model: dict[str, Any]
    schema: MotionFeatureSchema
    training: dict[str, Any]
    optimizer: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoint_version: int = MOTION_MAE_CHECKPOINT_VERSION
    checkpoint_type: str = "motion_mae"

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
    def from_dict(cls, payload: dict[str, Any]) -> "MotionMAECheckpointV1":
        if payload.get("checkpoint_version") != MOTION_MAE_CHECKPOINT_VERSION:
            raise ValueError("Unsupported Motion MAE checkpoint version.")
        if payload.get("checkpoint_type") != "motion_mae":
            raise ValueError("Expected checkpoint_type='motion_mae'.")
        return cls(
            meta=dict(payload["meta"]),
            model=dict(payload["model"]),
            schema=MotionFeatureSchema.from_dict(payload["schema"]),
            training=dict(payload["training"]),
            optimizer=dict(payload.get("optimizer", {})),
            artifacts=dict(payload.get("artifacts", {})),
        )


@dataclass(frozen=True)
class MotionMAEEncoderCheckpointV1:
    meta: dict[str, Any]
    model: dict[str, Any]
    schema: MotionFeatureSchema
    training: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoint_version: int = MOTION_MAE_CHECKPOINT_VERSION
    checkpoint_type: str = "motion_mae_encoder"

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
    def from_dict(cls, payload: dict[str, Any]) -> "MotionMAEEncoderCheckpointV1":
        if payload.get("checkpoint_version") != MOTION_MAE_CHECKPOINT_VERSION:
            raise ValueError("Unsupported Motion MAE encoder checkpoint version.")
        if payload.get("checkpoint_type") != "motion_mae_encoder":
            raise ValueError("Expected checkpoint_type='motion_mae_encoder'.")
        return cls(
            meta=dict(payload["meta"]),
            model=dict(payload["model"]),
            schema=MotionFeatureSchema.from_dict(payload["schema"]),
            training=dict(payload.get("training", {})),
            artifacts=dict(payload.get("artifacts", {})),
        )


def _model_kwargs_from_model(model: ReferenceMotionMAE) -> dict[str, Any]:
    return dict(model.model_kwargs())


def _build_training_payload(
    *,
    config: MotionMAEPretrainConfig,
    epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": config.to_dict(),
    }


def _extract_encoder_state_dict(model: ReferenceMotionMAE) -> dict[str, torch.Tensor]:
    encoder_keys = (
        "input_proj.",
        "encoder_position_embedding",
        "encoder.",
        "latent_norm.",
        "latent_proj.",
    )
    filtered: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key == "encoder_position_embedding" or key.startswith(encoder_keys):
            filtered[key] = value
    return filtered


def build_motion_mae_checkpoint(
    *,
    model: ReferenceMotionMAE,
    optimizer: torch.optim.Optimizer | None,
    schema: MotionFeatureSchema,
    config: MotionMAEPretrainConfig,
    epoch: int,
    best_metric: float,
    artifacts: dict[str, Any] | None = None,
) -> MotionMAECheckpointV1:
    return MotionMAECheckpointV1(
        meta={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "latent_dim": int(model.latent_dim),
            "model_kwargs": _model_kwargs_from_model(model),
        },
        model={"model": model.state_dict()},
        schema=schema,
        training=_build_training_payload(config=config, epoch=epoch, best_metric=best_metric),
        optimizer=optimizer.state_dict() if optimizer is not None else {},
        artifacts=artifacts or {},
    )


def build_motion_mae_encoder_checkpoint(
    *,
    model: ReferenceMotionMAE,
    schema: MotionFeatureSchema,
    config: MotionMAEPretrainConfig,
    epoch: int,
    best_metric: float,
    artifacts: dict[str, Any] | None = None,
) -> MotionMAEEncoderCheckpointV1:
    return MotionMAEEncoderCheckpointV1(
        meta={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "latent_dim": int(model.latent_dim),
            "model_kwargs": _model_kwargs_from_model(model),
            "frozen": True,
        },
        model={"encoder_state": _extract_encoder_state_dict(model)},
        schema=schema,
        training=_build_training_payload(config=config, epoch=epoch, best_metric=best_metric),
        artifacts=artifacts or {},
    )


def save_motion_mae_checkpoint(checkpoint: MotionMAECheckpointV1, path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_dict(), checkpoint_path)
    return checkpoint_path


def load_motion_mae_checkpoint(path: str | Path) -> MotionMAECheckpointV1:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Motion MAE checkpoint payload must be a dictionary.")
    return MotionMAECheckpointV1.from_dict(payload)


def save_motion_mae_encoder_checkpoint(checkpoint: MotionMAEEncoderCheckpointV1, path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_dict(), checkpoint_path)
    return checkpoint_path


def load_motion_mae_encoder_checkpoint(path: str | Path) -> MotionMAEEncoderCheckpointV1:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Motion MAE encoder checkpoint payload must be a dictionary.")
    return MotionMAEEncoderCheckpointV1.from_dict(payload)
