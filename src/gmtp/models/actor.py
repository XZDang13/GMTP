from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import torch
import torch.nn as nn
from RLAlg.nn.layers import GaussianHead, MLPLayer, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep
from RLAlg.normalizer import Normalizer

from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_policy_observation_spec,
    build_robot_policy_window_lengths,
)

from .film import FiLMResStack

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
DEFAULT_OBSERVATION_TERM_REGISTRY = _OBSERVATION_SPEC.DEFAULT_OBSERVATION_TERM_REGISTRY
ObservationLayout = _OBSERVATION_SPEC.ObservationLayout


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
    return max(block_ids, default=3)


@dataclass(frozen=True)
class RobotWindowTermLayout:
    term_id: str
    term_dim: int
    flat_slice: slice


@dataclass(frozen=True)
class RobotWindowLayout:
    window_length: int
    robot_obs_dim: int
    robot_step_dim: int
    terms: tuple[RobotWindowTermLayout, ...]


@lru_cache(maxsize=None)
def _build_robot_window_layout(action_dim: int, robot_window_length: int) -> RobotWindowLayout:
    if robot_window_length < 1:
        raise ValueError(f"robot_window_length must be positive, got {robot_window_length}.")

    spec = build_gmtp_policy_observation_spec(
        add_noise=False,
        window_lengths=build_robot_policy_window_lengths(robot_window_length),
    )
    layout = ObservationLayout(joint_dim=action_dim, action_dim=action_dim, key_body_count=0)
    robot_group = next(group for group in spec.enabled_groups() if group.name == "robot")

    offset = 0
    robot_step_dim = 0
    term_layouts: list[RobotWindowTermLayout] = []
    for term_spec in robot_group.terms:
        if not term_spec.enabled:
            continue
        if not term_spec.flatten and term_spec.window_length > 1:
            raise ValueError(
                "FiLMResActor only supports flattened robot observation history, "
                f"got term '{term_spec.id}' with flatten={term_spec.flatten} "
                f"and window_length={term_spec.window_length}."
            )
        if term_spec.window_length != robot_window_length:
            raise ValueError(
                "FiLMResActor requires a uniform robot window length, "
                f"got term '{term_spec.id}' with window_length={term_spec.window_length} "
                f"and expected {robot_window_length}."
            )

        term_dim = int(DEFAULT_OBSERVATION_TERM_REGISTRY[term_spec.type].dimension(layout, term_spec))
        flattened_dim = term_dim * term_spec.window_length if term_spec.flatten else term_dim
        term_layouts.append(
            RobotWindowTermLayout(
                term_id=term_spec.id,
                term_dim=term_dim,
                flat_slice=slice(offset, offset + flattened_dim),
            )
        )
        offset += flattened_dim
        robot_step_dim += term_dim

    return RobotWindowLayout(
        window_length=robot_window_length,
        robot_obs_dim=offset,
        robot_step_dim=robot_step_dim,
        terms=tuple(term_layouts),
    )


class FiLMResActor(nn.Module):
    def __init__(
        self,
        robot_obs_dim: int,
        motion_obs_dim: int,
        action_dim: int,
        num_blocks: int = 3,
        robot_window_length: int = 1,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if robot_window_length < 1:
            raise ValueError(f"robot_window_length must be positive, got {robot_window_length}.")

        self.robot_window_length = int(robot_window_length)
        self.robot_window_layout = _build_robot_window_layout(action_dim, self.robot_window_length)
        if robot_obs_dim != self.robot_window_layout.robot_obs_dim:
            raise ValueError(
                f"Expected robot_obs_dim={self.robot_window_layout.robot_obs_dim} for action_dim={action_dim} "
                f"and robot_window_length={self.robot_window_length}, got {robot_obs_dim}."
            )

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.num_blocks = num_blocks
        self.robot_step_dim = self.robot_window_layout.robot_step_dim
        self.robot_encoder = nn.Sequential(
            nn.Conv1d(self.robot_step_dim, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.robot_window_fuser = nn.Sequential(
            MLPLayer(256 * self.robot_window_length, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.Identity()),
        )
        self.motion_encoder = nn.Sequential(
            MLPLayer(motion_obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.Identity()),
        )
        self.stack = FiLMResStack(512, 512, num_layers=num_blocks)
        self.head = GaussianHead(512, action_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.stack.blocks

    def _reshape_robot_obs(self, robot_obs: torch.Tensor) -> torch.Tensor:
        if robot_obs.ndim != 2:
            raise ValueError(f"Expected robot_obs batch rank 2, got shape {tuple(robot_obs.shape)}.")

        batch_size = robot_obs.shape[0]
        robot_frames = []
        for term_layout in self.robot_window_layout.terms:
            term_flat = robot_obs[:, term_layout.flat_slice]
            robot_frames.append(term_flat.reshape(batch_size, self.robot_window_length, term_layout.term_dim))
        return torch.cat(robot_frames, dim=-1)

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        update_normlizer: bool = False,
    ) -> StochasticContinuousPolicyStep:
        robot_obs = self.robot_obs_normlizer(obs_dict["robot_obs"], update_normlizer)
        motion_obs = self.motion_obs_normlizer(obs_dict["motion_obs"], update_normlizer)
        robot_frames = self._reshape_robot_obs(robot_obs).transpose(1, 2)
        batch_size = robot_frames.shape[0]
        x_robot = self.robot_encoder(robot_frames)
        x_robot = self.robot_window_fuser(x_robot.reshape(batch_size, -1))
        x_motion = self.motion_encoder(motion_obs)
        x = self.stack(x_robot, x_motion)
        return self.head(x, action)


def build_actor(
    obs_dims: dict[str, int],
    actor_type: ActorType | str,
    action_dim: int,
    actor_kwargs: dict[str, int] | None = None,
) -> FiLMResActor:
    normalize_actor_type(str(actor_type))
    actor_kwargs = actor_kwargs or {}
    return FiLMResActor(
        obs_dims["robot"],
        obs_dims["motion"],
        action_dim,
        num_blocks=int(actor_kwargs.get("num_blocks", 3)),
        robot_window_length=int(actor_kwargs.get("robot_window_length", 1)),
    )


def get_actor_kwargs(
    actor: nn.Module,
    actor_type: ActorType | str,
) -> dict[str, int]:
    normalize_actor_type(str(actor_type))
    return {
        "num_blocks": int(actor.num_blocks),
        "robot_window_length": int(actor.robot_window_length),
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
) -> dict[str, tuple[int, ...]]:
    normalize_actor_type(str(actor_type))
    return {
        "motion_observations": (obs_dims["motion"],),
        "robot_observations": (obs_dims["robot"],),
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
