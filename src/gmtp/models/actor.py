from __future__ import annotations

import re
from enum import StrEnum

import torch
import torch.nn as nn
from RLAlg.nn.layers import GaussianHead, MLPLayer, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep
from RLAlg.normalizer import Normalizer

from .adain import BlockAttnResFiLMStack


class ActorType(StrEnum):
    FILM_ATTN_RES = "film_attn_res"


def normalize_actor_type(actor_type: str | None) -> ActorType:
    normalized = (actor_type or ActorType.FILM_ATTN_RES).lower().replace("-", "_")
    alias_map = {
        "film_attn_res": ActorType.FILM_ATTN_RES,
        "film_attnres": ActorType.FILM_ATTN_RES,
        "filmattnres": ActorType.FILM_ATTN_RES,
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported actor type '{actor_type}'. Only '{ActorType.FILM_ATTN_RES.value}' is supported."
        ) from exc


def infer_film_res_blocks(actor_weights: dict[str, torch.Tensor]) -> int:
    block_pattern = re.compile(r"^stack\.blocks\.(\d+)\.")
    block_ids = [
        int(match.group(1)) + 1
        for key in actor_weights
        if (match := block_pattern.match(key)) is not None
    ]
    return max(block_ids, default=3)


class FiLMAttnResActor(nn.Module):
    DEFAULT_ATTN_BLOCK_SIZE = BlockAttnResFiLMStack.DEFAULT_BLOCK_SIZE

    def __init__(
        self,
        robot_obs_dim: int,
        motion_obs_dim: int,
        action_dim: int,
        num_blocks: int = 3,
        attn_block_size: int = DEFAULT_ATTN_BLOCK_SIZE,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if attn_block_size < 1:
            raise ValueError(f"attn_block_size must be positive, got {attn_block_size}.")

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.num_blocks = num_blocks
        self.attn_block_size = attn_block_size
        self.robot_encoder = nn.Sequential(
            MLPLayer(robot_obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.Identity()),
        )
        self.motion_encoder = nn.Sequential(
            MLPLayer(motion_obs_dim, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.Identity()),
        )
        self.stack = BlockAttnResFiLMStack(512, 512, num_layers=num_blocks, block_size=attn_block_size)
        self.head = GaussianHead(512, action_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.stack.blocks

    @property
    def query_projs(self) -> nn.ModuleList:
        return self.stack.query_projs

    @property
    def attn_res(self):
        return self.stack.attn_res

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
    actor_kwargs: dict[str, int] | None = None,
) -> FiLMAttnResActor:
    normalize_actor_type(str(actor_type))
    actor_kwargs = actor_kwargs or {}
    return FiLMAttnResActor(
        obs_dims["robot"],
        obs_dims["motion"],
        action_dim,
        num_blocks=int(actor_kwargs.get("num_blocks", 3)),
        attn_block_size=int(actor_kwargs.get("attn_block_size", FiLMAttnResActor.DEFAULT_ATTN_BLOCK_SIZE)),
    )


def get_actor_kwargs(
    actor: nn.Module,
    actor_type: ActorType | str,
) -> dict[str, int]:
    normalize_actor_type(str(actor_type))
    return {
        "num_blocks": int(actor.num_blocks),
        "attn_block_size": int(actor.attn_block_size),
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
