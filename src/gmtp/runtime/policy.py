from __future__ import annotations

from pathlib import Path

import torch

from gmtp.models import (
    ActorType,
    build_actor,
    infer_film_res_blocks,
    normalize_actor_type,
)
from gmtp.models.motion_encoder import MotionEncoderType, normalize_motion_encoder_type
from gmtp.models.robot_encoder import RobotEncoderType, normalize_robot_encoder_type
from gmtp.runtime.checkpoints import CheckpointV2


def _resolve_existing_path(path: str | Path, *, label: str) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved_path}")
    return resolved_path


def resolve_motion_mae_checkpoint_path(
    checkpoint: CheckpointV2 | None = None,
    *,
    override: str | Path | None = None,
) -> Path | None:
    if override is not None:
        return _resolve_existing_path(override, label="Motion MAE encoder checkpoint")
    if checkpoint is None or checkpoint.motion_mae_encoder_checkpoint is None:
        return None
    return _resolve_existing_path(
        checkpoint.motion_mae_encoder_checkpoint,
        label="checkpoint Motion MAE encoder artifact",
    )


def resolve_checkpoint_actor_spec(
    checkpoint: CheckpointV2,
    *,
    actor_type_override: str | None = None,
    num_blocks: int | None = None,
    motion_encoder_type_override: str | None = None,
) -> tuple[ActorType, dict[str, int | str]]:
    actor_type = normalize_actor_type(actor_type_override or checkpoint.meta.get("actor_type"))
    actor_weights = checkpoint.model["actor"]
    checkpoint_actor_kwargs = dict(checkpoint.meta.get("actor_kwargs", {}))
    robot_window_length = int(checkpoint_actor_kwargs.get("robot_window_length", 1))
    requested_robot_encoder_type = checkpoint_actor_kwargs.get(
        "robot_encoder_type",
        RobotEncoderType.TRANSFORMER.value,
    )
    motion_window_length = int(checkpoint_actor_kwargs.get("motion_window_length", 1))
    requested_motion_encoder_type = checkpoint_actor_kwargs.get(
        "motion_encoder_type",
        motion_encoder_type_override or MotionEncoderType.TRANSFORMER.value,
    )
    actor_kwargs = {
        "num_blocks": int(
            num_blocks
            if num_blocks is not None
            else checkpoint_actor_kwargs.get("num_blocks", infer_film_res_blocks(actor_weights))
        ),
        "robot_window_length": robot_window_length,
        "robot_encoder_type": str(
            RobotEncoderType.MLP
            if robot_window_length == 1
            else normalize_robot_encoder_type(requested_robot_encoder_type)
        ),
        "motion_window_length": motion_window_length,
        "motion_encoder_type": str(
            MotionEncoderType.MLP
            if motion_window_length == 1
            else normalize_motion_encoder_type(motion_encoder_type_override or requested_motion_encoder_type)
        ),
    }
    return actor_type, actor_kwargs


def load_actor_from_checkpoint(
    checkpoint: CheckpointV2,
    *,
    obs_dims: dict[str, int],
    action_dim: int,
    device: torch.device,
    actor_type_override: str | None = None,
    num_blocks: int | None = None,
    motion_encoder_type_override: str | None = None,
    motion_mae_encoder_checkpoint: str | Path | None = None,
) -> tuple[torch.nn.Module, ActorType, dict[str, int | str]]:
    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(
        checkpoint,
        actor_type_override=actor_type_override,
        num_blocks=num_blocks,
        motion_encoder_type_override=motion_encoder_type_override,
    )
    actor = build_actor(
        obs_dims,
        actor_type,
        action_dim,
        actor_kwargs=actor_kwargs,
        motion_mae_encoder_checkpoint=motion_mae_encoder_checkpoint,
        device=device,
    ).to(device)
    actor.load_state_dict(checkpoint.model["actor"])
    actor.eval()
    return actor, actor_type, actor_kwargs


def resolve_checkpoint_stem(path: str | Path) -> str:
    return Path(path).expanduser().resolve().stem


def validate_checkpoint_actor_observation_dims(
    checkpoint: CheckpointV2,
    *,
    checkpoint_obs_dims: dict[str, int],
    runtime_obs_dims: dict[str, int],
    motion_mae_encoder_checkpoint: str | Path | None = None,
) -> None:
    if (
        checkpoint_obs_dims["motion"] == runtime_obs_dims["motion"]
        and checkpoint_obs_dims["robot"] == runtime_obs_dims["robot"]
    ):
        return

    checkpoint_actor_kwargs = dict(checkpoint.meta.get("actor_kwargs", {}))
    motion_window_length = int(checkpoint_actor_kwargs.get("motion_window_length", 1))
    requested_motion_encoder_type = checkpoint_actor_kwargs.get(
        "motion_encoder_type",
        MotionEncoderType.TRANSFORMER.value,
    )
    uses_integrated_motion_mae = (
        motion_window_length > 1
        and normalize_motion_encoder_type(requested_motion_encoder_type) == MotionEncoderType.MAE
    )
    if motion_mae_encoder_checkpoint is not None and not uses_integrated_motion_mae:
        raise ValueError(
            "Checkpoint actor observation dims do not match runtime env dims because this checkpoint depends on "
            "the removed Motion MAE latent-append adapter path. Re-export or retrain with "
            "motion_window_length>1 and motion_encoder_type='mae'."
        )

    raise ValueError(
        "Checkpoint actor observation dims do not match runtime env dims: "
        f"checkpoint={checkpoint_obs_dims}, runtime={runtime_obs_dims}."
    )
