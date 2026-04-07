from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gmtp.motion_mae import build_frozen_motion_mae_encoder
from gmtp.motion_mae.features import quat_apply_inverse
from gmtp.motion_mae.policy import FrozenMotionMAEEncoder
from gmtp.motion_mae.schema import MotionFeatureSchema
from gmtp.models import (
    ActorType,
    build_actor,
    infer_film_res_blocks,
    normalize_actor_type,
)
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
) -> tuple[ActorType, dict[str, int | str]]:
    actor_type = normalize_actor_type(actor_type_override or checkpoint.meta.get("actor_type"))
    actor_weights = checkpoint.model["actor"]
    checkpoint_actor_kwargs = dict(checkpoint.meta.get("actor_kwargs", {}))
    robot_window_length = int(checkpoint_actor_kwargs.get("robot_window_length", 1))
    requested_robot_encoder_type = checkpoint_actor_kwargs.get(
        "robot_encoder_type",
        RobotEncoderType.TRANSFORMER.value,
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
) -> tuple[torch.nn.Module, ActorType, dict[str, int | str]]:
    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(
        checkpoint,
        actor_type_override=actor_type_override,
        num_blocks=num_blocks,
    )
    actor = build_actor(obs_dims, actor_type, action_dim, actor_kwargs=actor_kwargs).to(device)
    actor.load_state_dict(checkpoint.model["actor"])
    actor.eval()
    return actor, actor_type, actor_kwargs


def resolve_checkpoint_stem(path: str | Path) -> str:
    return Path(path).expanduser().resolve().stem


class MotionMAELatentAdapter:
    def __init__(
        self,
        frozen_encoder: FrozenMotionMAEEncoder,
        *,
        checkpoint_path: str | Path,
    ) -> None:
        self.encoder = frozen_encoder
        self.schema = frozen_encoder.schema
        self.latent_dim = int(frozen_encoder.latent_dim)
        self.window_length = int(frozen_encoder.encoder.past_frames)
        self.policy_motion_dim = int(self.schema.policy_motion_slice.dim)
        self.device = frozen_encoder.reference_mean.device
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.gravity_vector = torch.as_tensor(
            self.schema.gravity_vector,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3)
        self._history: torch.Tensor | None = None
        self._body_index_cache: dict[tuple[str, ...], tuple[int, tuple[int, ...]]] = {}

    def augment_observation_dims(self, obs_dims: dict[str, int]) -> dict[str, int]:
        augmented_dims = dict(obs_dims)
        augmented_dims["motion"] = int(obs_dims["motion"]) + self.latent_dim
        augmented_dims["policy"] = int(obs_dims["policy"]) + self.latent_dim
        return augmented_dims

    def initialize_history(self, env: Any) -> None:
        current_frame = self._extract_reference_frame(env)
        self._history = current_frame.unsqueeze(1).repeat(1, self.window_length, 1)

    def update_history(self, env: Any, *, done: torch.Tensor | None = None) -> None:
        current_frame = self._extract_reference_frame(env)
        if self._history is None or self._history.shape[0] != current_frame.shape[0]:
            self.initialize_history(env)
            return

        updated_history = torch.cat((self._history[:, 1:], current_frame.unsqueeze(1)), dim=1)
        if done is not None:
            done_mask = torch.as_tensor(done, dtype=torch.bool, device=self.device).reshape(-1)
            if done_mask.numel() != updated_history.shape[0]:
                raise ValueError(
                    f"Expected done mask batch {updated_history.shape[0]}, got {done_mask.numel()}."
                )
            if bool(done_mask.any().item()):
                updated_history = updated_history.clone()
                repeated = current_frame[done_mask].unsqueeze(1).repeat(1, self.window_length, 1)
                updated_history[done_mask] = repeated
        self._history = updated_history

    @property
    def history(self) -> torch.Tensor:
        if self._history is None:
            raise RuntimeError("MotionMAELatentAdapter history is uninitialized.")
        return self._history

    def augment_actor_observation(self, actor_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion_obs = torch.as_tensor(actor_obs["motion_obs"], dtype=torch.float32, device=self.device)
        robot_obs = torch.as_tensor(actor_obs["robot_obs"], dtype=torch.float32, device=self.device)
        if motion_obs.ndim != 2:
            raise ValueError(f"Expected policy motion observation rank 2, got shape {tuple(motion_obs.shape)}.")
        if motion_obs.shape[-1] != self.policy_motion_dim:
            raise ValueError(
                f"Expected policy motion observation dim {self.policy_motion_dim}, got {motion_obs.shape[-1]}."
            )

        latent = self.encoder(self.history)
        if latent.shape[0] != motion_obs.shape[0]:
            raise ValueError(f"Expected latent batch {motion_obs.shape[0]}, got {latent.shape[0]}.")

        augmented_obs = dict(actor_obs)
        augmented_obs["motion_obs"] = torch.cat((motion_obs, latent), dim=-1)
        augmented_obs["robot_obs"] = robot_obs
        return augmented_obs

    def _extract_reference_frame(self, env: Any) -> torch.Tensor:
        base_env = getattr(env, "unwrapped", env)
        reference_motion = getattr(base_env, "reference_motion", None)
        if reference_motion is not None:
            return self._build_reference_features(
                joint_pos=getattr(reference_motion, "joint_pos"),
                joint_vel=getattr(reference_motion, "joint_vel"),
                body_positions=getattr(reference_motion, "body_positions"),
                body_quaternions=getattr(reference_motion, "body_quaternions"),
                anchor_body_index=int(getattr(reference_motion, "anchor_body_index")),
                body_names=getattr(getattr(base_env, "motion_lib", None), "body_names", None),
            )

        motion_lib = getattr(base_env, "motion_lib", None)
        motion_id = getattr(base_env, "motion_id", None)
        times = getattr(base_env, "times", None)
        anchor_body_index = getattr(base_env, "anchor_body_index", None)
        if motion_lib is None or motion_id is None or times is None or anchor_body_index is None:
            raise RuntimeError(
                "The current environment does not expose reference motion tensors required for Motion MAE startup."
            )

        motion_ids = torch.as_tensor(motion_id, dtype=torch.long).reshape(-1)
        query_times = torch.as_tensor(times, dtype=torch.float32).reshape(-1)
        if motion_ids.numel() == 1 and query_times.numel() > 1:
            motion_ids = motion_ids.expand(query_times.numel())
        sampled_motion = motion_lib.sample_motion(motion_ids=motion_ids, times=query_times)
        return self._build_reference_features(
            joint_pos=sampled_motion["joint_pos"],
            joint_vel=sampled_motion["joint_vel"],
            body_positions=sampled_motion["body_positions"],
            body_quaternions=sampled_motion["body_quaternions"],
            anchor_body_index=int(anchor_body_index),
            body_names=getattr(motion_lib, "body_names", None),
        )

    def _resolve_body_indices(self, body_names: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
        cached = self._body_index_cache.get(body_names)
        if cached is not None:
            return cached

        try:
            anchor_body_index = body_names.index(self.schema.anchor_body_name)
        except ValueError as exc:
            raise ValueError(
                f"Anchor body '{self.schema.anchor_body_name}' is missing from runtime body_names."
            ) from exc

        try:
            end_effector_body_indices = tuple(body_names.index(name) for name in self.schema.end_effector_body_names)
        except ValueError as exc:
            raise ValueError(
                "One or more runtime end-effector bodies are missing from runtime body_names: "
                f"{self.schema.end_effector_body_names}."
            ) from exc

        self._body_index_cache[body_names] = (anchor_body_index, end_effector_body_indices)
        return anchor_body_index, end_effector_body_indices

    def _as_batched_tensor(
        self,
        value: Any,
        *,
        name: str,
        trailing_rank: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == trailing_rank:
            tensor = tensor.unsqueeze(0)
        expected_rank = trailing_rank + 1
        if tensor.ndim != expected_rank:
            raise ValueError(f"Expected {name} rank {expected_rank}, got shape {tuple(tensor.shape)}.")
        return tensor

    def _build_reference_features(
        self,
        *,
        joint_pos: Any,
        joint_vel: Any,
        body_positions: Any,
        body_quaternions: Any,
        anchor_body_index: int,
        body_names: Any,
    ) -> torch.Tensor:
        joint_pos_tensor = self._as_batched_tensor(joint_pos, name="joint_pos", trailing_rank=1)
        joint_vel_tensor = self._as_batched_tensor(joint_vel, name="joint_vel", trailing_rank=1)
        body_positions_tensor = self._as_batched_tensor(body_positions, name="body_positions", trailing_rank=2)
        body_quaternions_tensor = self._as_batched_tensor(body_quaternions, name="body_quaternions", trailing_rank=2)

        if joint_pos_tensor.shape != joint_vel_tensor.shape:
            raise ValueError("Runtime joint_pos and joint_vel tensors must have matching shapes.")
        if body_positions_tensor.shape[:-1] != body_quaternions_tensor.shape[:-1]:
            raise ValueError("Runtime body_positions and body_quaternions tensors must have matching batch/body shapes.")

        resolved_body_names = tuple(str(item) for item in (body_names or self.schema.body_names))
        if resolved_body_names and len(resolved_body_names) != body_positions_tensor.shape[1]:
            raise ValueError(
                "Runtime body_names length does not match reference body tensor shape: "
                f"{len(resolved_body_names)} vs {body_positions_tensor.shape[1]}."
            )

        if resolved_body_names:
            expected_anchor_index, end_effector_indices = self._resolve_body_indices(resolved_body_names)
            if expected_anchor_index != anchor_body_index:
                raise ValueError(
                    f"Runtime anchor_body_index mismatch: expected {expected_anchor_index}, got {anchor_body_index}."
                )
        else:
            raise ValueError("Motion MAE startup requires body_names from the checkpoint schema or runtime env.")

        if self.schema.joint_names and len(self.schema.joint_names) != joint_pos_tensor.shape[-1]:
            raise ValueError(
                "Runtime joint dimension does not match Motion MAE schema: "
                f"{joint_pos_tensor.shape[-1]} vs {len(self.schema.joint_names)}."
            )

        batch_size = joint_pos_tensor.shape[0]
        gravity = self.gravity_vector.expand(batch_size, -1)
        anchor_positions = body_positions_tensor[:, anchor_body_index]
        anchor_quaternions = body_quaternions_tensor[:, anchor_body_index]
        root_features = quat_apply_inverse(anchor_quaternions, gravity)
        joint_features = torch.cat((joint_pos_tensor, joint_vel_tensor), dim=-1)
        end_effector_positions = body_positions_tensor[:, end_effector_indices]
        rel_end_effector_positions = quat_apply_inverse(
            anchor_quaternions[:, None, :].expand(-1, len(end_effector_indices), -1),
            end_effector_positions - anchor_positions[:, None, :],
        ).reshape(batch_size, -1)
        full_features = torch.cat((root_features, joint_features, rel_end_effector_positions), dim=-1)

        base_slice_map = {item.name: item for item in self.schema.base_slices}
        reference_features = torch.cat(
            [
                full_features[:, base_slice_map[name].start : base_slice_map[name].end]
                for name in self.schema.reference_feature_names
            ],
            dim=-1,
        )
        if reference_features.shape[-1] != self.schema.d_ref:
            raise ValueError(f"Expected reference feature dim {self.schema.d_ref}, got {reference_features.shape[-1]}.")
        return reference_features


def build_motion_mae_adapter(
    path: str | Path | None,
    *,
    device: torch.device | str,
) -> MotionMAELatentAdapter | None:
    if path is None:
        return None
    checkpoint_path = _resolve_existing_path(path, label="Motion MAE encoder checkpoint")
    frozen_encoder = build_frozen_motion_mae_encoder(checkpoint_path, device=device)
    return MotionMAELatentAdapter(frozen_encoder, checkpoint_path=checkpoint_path)
