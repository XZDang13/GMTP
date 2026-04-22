from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from gmtp.integrations.ref2act.motion import motion_label, motion_names, resolve_motion_files
from gmtp.integrations.ref2act.observation_history import normalize_observation_window_lengths
from gmtp.models import ActorType, get_actor_kwargs

CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class CheckpointV2:
    meta: dict[str, Any]
    model: dict[str, Any]
    env: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    checkpoint_version: int = CHECKPOINT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointV2":
        version = payload.get("checkpoint_version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"Expected checkpoint_version={CHECKPOINT_VERSION}, got {version!r}.")
        required_keys = ("meta", "model", "env", "artifacts")
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise KeyError(f"CheckpointV2 is missing required keys: {missing}.")
        return cls(
            meta=dict(payload["meta"]),
            model=dict(payload["model"]),
            env=dict(payload["env"]),
            artifacts=dict(payload["artifacts"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "meta": self.meta,
            "model": self.model,
            "env": self.env,
            "artifacts": self.artifacts,
        }

    @property
    def actor_type(self) -> str:
        return str(self.meta["actor_type"])

    @property
    def actor_kwargs(self) -> dict[str, Any]:
        return dict(self.meta.get("actor_kwargs", {}))

    @property
    def motion_files(self) -> list[str]:
        return list(self.env.get("motion_files", []))

    @property
    def observation_window_lengths(self) -> dict[str, int]:
        return normalize_observation_window_lengths(self.env.get("observation_window_lengths"))

    @property
    def motion_mae_encoder_checkpoint(self) -> str | None:
        value = self.artifacts.get("motion_mae_encoder_checkpoint")
        if value is None:
            return None
        return str(Path(value).expanduser().resolve())


def load_checkpoint_v2(path: str | Path) -> CheckpointV2:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a dictionary payload.")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint at {checkpoint_path} is not CheckpointV2.")
    return CheckpointV2.from_dict(payload)


def save_checkpoint_v2(checkpoint: CheckpointV2, path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_dict(), checkpoint_path)
    return checkpoint_path


def build_training_checkpoint(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    motion_files: list[str],
    joint_params: dict[str, Any],
    action_mode: str | None,
    root_name: str | None,
    anchor_body_name: str | None,
    segment_source: str | None = None,
    sampling_strategy: str | None = None,
    motion_mae_encoder_checkpoint: str | None = None,
    observation_window_lengths: dict[str, int] | None = None,
    artifacts: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> CheckpointV2:
    resolved_motion_files = resolve_motion_files(motion_files)
    resolved_window_lengths = normalize_observation_window_lengths(observation_window_lengths)
    env_payload = {
        "motion_files": resolved_motion_files,
        "motion_names": motion_names(resolved_motion_files),
        "joint_names": joint_params["joint_names"],
        "joint_effort_limits": joint_params["joint_effort_limits"],
        "joint_pos_limits": joint_params["joint_pos_limits"],
        "joint_stiffness": joint_params["joint_stiffness"],
        "joint_damping": joint_params["joint_damping"],
        "action_offset": joint_params["action_offset"],
        "action_scale": joint_params["action_scale"],
        "action_mode": action_mode,
        "root_name": root_name,
        "anchor_body_name": anchor_body_name,
    }
    if segment_source is not None:
        env_payload["segment_source"] = segment_source
    if sampling_strategy is not None:
        env_payload["sampling_strategy"] = sampling_strategy
    if observation_window_lengths is not None:
        env_payload["observation_window_lengths"] = resolved_window_lengths

    checkpoint_artifacts = dict(artifacts or {})
    if motion_mae_encoder_checkpoint is not None:
        checkpoint_artifacts["motion_mae_encoder_checkpoint"] = str(
            Path(motion_mae_encoder_checkpoint).expanduser().resolve()
        )

    return CheckpointV2(
        meta={
            "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
            "actor_type": ActorType.FILM_RES.value,
            "actor_kwargs": get_actor_kwargs(actor, ActorType.FILM_RES),
            "motion_label": motion_label(resolved_motion_files),
        },
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env=env_payload,
        artifacts=checkpoint_artifacts,
    )
