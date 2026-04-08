from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

import torch
import torch.nn as nn
from RLAlg.nn.layers import GaussianHead
from RLAlg.nn.steps import StochasticContinuousPolicyStep
from RLAlg.normalizer import Normalizer

from .film import FiLMResStack
from .motion_encoder import (
    MotionEncoderType,
    MotionHistoryEncoder,
    build_motion_window_layout,
    normalize_motion_encoder_type,
    reshape_motion_history,
)
from .robot_encoder import (
    RobotHistoryEncoder,
    RobotEncoderType,
    build_robot_window_layout,
    normalize_robot_encoder_type,
    reshape_robot_history,
)

ACTOR_HIDDEN_DIM = 512


class ActorType(StrEnum):
    FILM_RES = "film_res"


def normalize_actor_type(actor_type: str | None) -> ActorType:
    normalized = (actor_type or ActorType.FILM_RES).lower().replace("-", "_")
    alias_map = {
        "film_res": ActorType.FILM_RES,
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported actor type '{actor_type}'. Only '{ActorType.FILM_RES.value}' is supported."
        ) from exc


def infer_film_res_blocks(actor_weights: dict[str, torch.Tensor]) -> int:
    block_pattern = re.compile(r"^stack\.blocks\.(\d+)\.")
    block_ids = [
        int(match.group(1)) + 1
        for key in actor_weights
        if (match := block_pattern.match(key)) is not None
    ]
    return max(block_ids, default=4)


class FiLMResActor(nn.Module):
    def __init__(
        self,
        robot_obs_dim: int,
        motion_obs_dim: int,
        action_dim: int,
        num_blocks: int = 4,
        robot_window_length: int = 1,
        robot_encoder_type: str | RobotEncoderType = RobotEncoderType.TRANSFORMER,
        motion_window_length: int = 1,
        motion_encoder_type: str | MotionEncoderType = MotionEncoderType.TRANSFORMER,
        motion_mae_encoder_checkpoint: str | Path | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if robot_window_length < 1:
            raise ValueError(f"robot_window_length must be positive, got {robot_window_length}.")
        if motion_window_length < 1:
            raise ValueError(f"motion_window_length must be positive, got {motion_window_length}.")

        self.robot_window_length = int(robot_window_length)
        self.robot_window_layout = build_robot_window_layout(action_dim, self.robot_window_length)
        robot_normalizer_shape = (
            (self.robot_window_layout.window_length, self.robot_window_layout.robot_step_dim)
            if self.robot_window_length > 1
            else (robot_obs_dim,)
        )
        self.motion_window_length = int(motion_window_length)
        self.motion_window_layout = build_motion_window_layout(action_dim, self.motion_window_length)
        motion_normalizer_shape = (
            (self.motion_window_layout.window_length, self.motion_window_layout.motion_step_dim)
            if self.motion_window_length > 1
            else (motion_obs_dim,)
        )
        self.robot_obs_normlizer = Normalizer(robot_normalizer_shape)
        self.motion_obs_normlizer = Normalizer(motion_normalizer_shape)
        self.num_blocks = num_blocks
        self.robot_step_dim = self.robot_window_layout.robot_step_dim
        self.motion_step_dim = self.motion_window_layout.motion_step_dim
        self.robot_encoder_type = (
            RobotEncoderType.MLP
            if self.robot_window_length == 1
            else normalize_robot_encoder_type(robot_encoder_type)
        )
        self.motion_encoder_type = (
            MotionEncoderType.MLP
            if self.motion_window_length == 1
            else normalize_motion_encoder_type(motion_encoder_type)
        )
        self.robot_encoder = RobotHistoryEncoder(
            robot_obs_dim=robot_obs_dim,
            action_dim=action_dim,
            robot_window_length=self.robot_window_length,
            robot_encoder_type=self.robot_encoder_type,
        )
        self.motion_encoder = MotionHistoryEncoder(
            motion_obs_dim=motion_obs_dim,
            action_dim=action_dim,
            motion_window_length=self.motion_window_length,
            motion_encoder_type=self.motion_encoder_type,
            motion_mae_encoder_checkpoint=motion_mae_encoder_checkpoint,
            device=device,
        )
        self.stack = FiLMResStack(ACTOR_HIDDEN_DIM, ACTOR_HIDDEN_DIM, num_layers=num_blocks)
        self.head = GaussianHead(ACTOR_HIDDEN_DIM, action_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.stack.blocks

    def _reshape_robot_obs(self, robot_obs: torch.Tensor) -> torch.Tensor:
        return reshape_robot_history(robot_obs, self.robot_window_layout)

    def _reshape_motion_obs(self, motion_obs: torch.Tensor) -> torch.Tensor:
        return reshape_motion_history(motion_obs, self.motion_window_layout)

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        update_normlizer: bool = False,
    ) -> StochasticContinuousPolicyStep:
        robot_obs = self.robot_obs_normlizer(obs_dict["robot_obs"], update_normlizer)
        motion_obs = self.motion_obs_normlizer(obs_dict["motion_obs"], update_normlizer)
        x_robot = self.robot_encoder(robot_obs)
        x_motion = self.motion_encoder(motion_obs)
        x = self.stack(x_robot, x_motion)
        return self.head(x, action)


def build_actor(
    obs_dims: dict[str, int],
    actor_type: ActorType | str,
    action_dim: int,
    actor_kwargs: dict[str, int | str] | None = None,
    *,
    motion_mae_encoder_checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> FiLMResActor:
    normalize_actor_type(str(actor_type))
    actor_kwargs = actor_kwargs or {}
    return FiLMResActor(
        obs_dims["robot"],
        obs_dims["motion"],
        action_dim,
        num_blocks=int(actor_kwargs.get("num_blocks", 4)),
        robot_window_length=int(actor_kwargs.get("robot_window_length", 1)),
        robot_encoder_type=str(actor_kwargs.get("robot_encoder_type", RobotEncoderType.TRANSFORMER.value)),
        motion_window_length=int(actor_kwargs.get("motion_window_length", 1)),
        motion_encoder_type=str(actor_kwargs.get("motion_encoder_type", MotionEncoderType.TRANSFORMER.value)),
        motion_mae_encoder_checkpoint=motion_mae_encoder_checkpoint,
        device=device,
    )


def get_actor_kwargs(
    actor: nn.Module,
    actor_type: ActorType | str,
) -> dict[str, int | str]:
    normalize_actor_type(str(actor_type))
    return {
        "num_blocks": int(actor.num_blocks),
        "robot_window_length": int(actor.robot_window_length),
        "robot_encoder_type": str(actor.robot_encoder_type),
        "motion_window_length": int(actor.motion_window_length),
        "motion_encoder_type": str(actor.motion_encoder_type),
    }


def get_actor_observation(
    obs: dict[str, torch.Tensor],
    actor_type: ActorType | str,
) -> dict[str, torch.Tensor]:
    normalize_actor_type(str(actor_type))
    return {
        "motion_obs": obs["motion"],
        "robot_obs": obs["robot"],
    }


def get_policy_storage_specs(
    obs_dims: dict[str, int],
    actor_type: ActorType | str,
    actor_kwargs: dict[str, int | str] | None = None,
) -> dict[str, tuple[int, ...]]:
    normalize_actor_type(str(actor_type))
    actor_kwargs = actor_kwargs or {}
    motion_window_length = int(actor_kwargs.get("motion_window_length", 1))
    robot_window_length = int(actor_kwargs.get("robot_window_length", 1))
    if motion_window_length > 1:
        if obs_dims["motion"] % motion_window_length != 0:
            raise ValueError(
                f"Motion observation dim {obs_dims['motion']} is not divisible by motion_window_length={motion_window_length}."
            )
        motion_storage_shape = (motion_window_length, obs_dims["motion"] // motion_window_length)
    else:
        motion_storage_shape = (obs_dims["motion"],)
    if robot_window_length > 1:
        if obs_dims["robot"] % robot_window_length != 0:
            raise ValueError(
                f"Robot observation dim {obs_dims['robot']} is not divisible by robot_window_length={robot_window_length}."
            )
        robot_storage_shape = (robot_window_length, obs_dims["robot"] // robot_window_length)
    else:
        robot_storage_shape = (obs_dims["robot"],)
    return {
        "motion_observations": motion_storage_shape,
        "robot_observations": robot_storage_shape,
    }


def get_policy_records(
    actor_obs: dict[str, torch.Tensor],
    actor_type: ActorType | str,
) -> dict[str, torch.Tensor]:
    normalize_actor_type(str(actor_type))
    return {
        "motion_observations": actor_obs["motion_obs"],
        "robot_observations": actor_obs["robot_obs"],
    }


def get_policy_batch(
    batch: dict[str, torch.Tensor],
    actor_type: ActorType | str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    normalize_actor_type(str(actor_type))
    return {
        "motion_obs": batch["motion_observations"].to(device),
        "robot_obs": batch["robot_observations"].to(device),
    }
