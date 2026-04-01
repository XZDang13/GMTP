from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from gmtp.integrations.ref2act.motion import motion_label, motion_names, resolve_motion_files
from gmtp.models import get_actor_kwargs, normalize_actor_type

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
    def actor_kwargs(self) -> dict[str, int]:
        return dict(self.meta.get("actor_kwargs", {}))

    @property
    def motion_files(self) -> list[str]:
        return list(self.env.get("motion_files", []))


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
    actor_type: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    motion_files: list[str],
    joint_params: dict[str, Any],
    action_mode: str | None,
    root_name: str | None,
    anchor_body_name: str | None,
    artifacts: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> CheckpointV2:
    normalized_actor_type = normalize_actor_type(actor_type)
    resolved_motion_files = resolve_motion_files(motion_files)
    return CheckpointV2(
        meta={
            "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
            "actor_type": normalized_actor_type.value,
            "actor_kwargs": get_actor_kwargs(actor, normalized_actor_type),
            "motion_label": motion_label(resolved_motion_files),
        },
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={
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
        },
        artifacts=artifacts or {},
    )
