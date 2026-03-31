from __future__ import annotations

import re
from enum import StrEnum

import torch
import torch.nn as nn

from RLAlg.normalizer import Normalizer
from RLAlg.nn.layers import GRULayer, GaussianHead, MLPLayer, NormPosition
from RLAlg.nn.steps import StochasticContinuousPolicyStep

from .adain import AdaINBlock, AdaINResBlock


def _normalize_observation(
    normalizer: Normalizer,
    obs: torch.Tensor,
    update_normalizer: bool = False,
) -> torch.Tensor:
    if obs.ndim == 2:
        return normalizer(obs, update=update_normalizer)
    if obs.ndim == 3:
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_obs = normalizer(flat_obs, update=update_normalizer)
        return flat_obs.reshape(*obs.shape)
    raise ValueError(f"Expected observation rank 2 or 3, got shape {tuple(obs.shape)}.")


class ActorType(StrEnum):
    VANILA = "vanila"
    RECURRENT = "recurrent"
    SPLIT_ENCODER = "split_encoder"
    ADAIN = "adain"
    ADAIN_RES = "adain_res"


def normalize_actor_type(actor_type: str | None) -> ActorType:
    normalized = (actor_type or ActorType.VANILA).lower().replace("-", "_")
    alias_map = {
        "vanila": ActorType.VANILA,
        "vanilla": ActorType.VANILA,
        "recurrent": ActorType.RECURRENT,
        "gru": ActorType.RECURRENT,
        "vanila_gru": ActorType.RECURRENT,
        "vanilla_gru": ActorType.RECURRENT,
        "split": ActorType.SPLIT_ENCODER,
        "split_encoder": ActorType.SPLIT_ENCODER,
        "adain": ActorType.ADAIN,
        "adain_res": ActorType.ADAIN_RES,
        "adainres": ActorType.ADAIN_RES,
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported actor type '{actor_type}'.") from exc


def is_concat_actor(actor_type: ActorType | str) -> bool:
    normalized = normalize_actor_type(str(actor_type))
    return normalized in {ActorType.VANILA, ActorType.RECURRENT}


def is_recurrent_actor(actor_type: ActorType | str) -> bool:
    return normalize_actor_type(str(actor_type)) == ActorType.RECURRENT


def unpack_actor_output(actor_output):
    if isinstance(actor_output, tuple):
        if len(actor_output) != 2:
            raise ValueError(
                f"Expected recurrent actor output to be (step, next_state), got tuple of length {len(actor_output)}."
            )
        return actor_output
    return actor_output, None


def policy_state_for_storage(policy_state: torch.Tensor) -> torch.Tensor:
    return policy_state.transpose(0, 1)


def policy_state_from_storage(policy_state: torch.Tensor, device: torch.device) -> torch.Tensor:
    return policy_state.to(device).transpose(0, 1).contiguous()


def infer_adain_res_blocks(actor_weights: dict[str, torch.Tensor]) -> int:
    block_pattern = re.compile(r"^block_(\d+)\.")
    block_ids = [
        int(match.group(1))
        for key in actor_weights
        if (match := block_pattern.match(key)) is not None
    ]
    return max(block_ids, default=3)


def infer_recurrent_actor_kwargs(actor_weights: dict[str, torch.Tensor]) -> dict[str, int]:
    layer_pattern = re.compile(r"^gru\.gru\.weight_ih_l(\d+)$")
    layer_ids = [
        int(match.group(1))
        for key in actor_weights
        if (match := layer_pattern.match(key)) is not None
    ]
    if not layer_ids:
        raise ValueError("Could not infer recurrent actor configuration from checkpoint weights.")

    hidden_size = int(actor_weights["gru.gru.weight_hh_l0"].shape[1])
    return {
        "hidden_size": hidden_size,
        "num_layers": max(layer_ids) + 1,
    }


class VanilaActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.normlizer = Normalizer((obs_dim,))
        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 2048, nn.SiLU(), NormPosition.POST),
            MLPLayer(2048, 1024, nn.SiLU(), NormPosition.POST),
            MLPLayer(1024, 512, nn.SiLU(), NormPosition.POST),
        )
        self.head = GaussianHead(512, action_dim)

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        update_normlizer: bool = False,
    ) -> StochasticContinuousPolicyStep:
        obs = obs_dict["obs"]
        obs = self.normlizer(obs, update=update_normlizer)
        x = self.encoder(obs)
        return self.head(x, action)


class RecurrentActor(nn.Module):
    DEFAULT_HIDDEN_SIZE = 512
    DEFAULT_NUM_LAYERS = 1

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ):
        super().__init__()

        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.normlizer = Normalizer((obs_dim,))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 2048, nn.SiLU(), NormPosition.POST),
            MLPLayer(2048, 1024, nn.SiLU(), NormPosition.POST),
            MLPLayer(1024, 512, nn.SiLU(), NormPosition.POST),
        )
        self.gru = GRULayer(512, hidden_size, num_layers=num_layers)
        self.head = GaussianHead(hidden_size, action_dim)

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        ref_param = self.head.mu_layer.weight
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device or ref_param.device,
            dtype=dtype or ref_param.dtype,
        )

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        episode_starts: torch.Tensor | None = None,
        update_normlizer: bool = False,
    ) -> tuple[StochasticContinuousPolicyStep, torch.Tensor]:
        obs = obs_dict["obs"]
        obs = _normalize_observation(self.normlizer, obs, update_normlizer)
        x = self.encoder(obs)
        x, next_state = self.gru(x, hidden_state=initial_state, episode_starts=episode_starts)
        step = self.head(x, action)
        return step, next_state


class SplitEncoderActor(nn.Module):
    def __init__(self, robot_obs_dim: int, motion_obs_dim: int, action_dim: int):
        super().__init__()

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.robot_encoder = MLPLayer(robot_obs_dim, 256, nn.Identity())
        self.motion_encoder = MLPLayer(motion_obs_dim, 256, nn.Identity())
        self.encoder = nn.Sequential(
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
            MLPLayer(512, 512, nn.SiLU(), NormPosition.POST),
        )
        self.head = GaussianHead(512, action_dim)

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
        x = torch.cat([x_robot, x_motion], dim=-1)
        x = self.encoder(x)
        return self.head(x, action)


class AdaINActor(nn.Module):
    def __init__(self, robot_obs_dim: int, motion_obs_dim: int, action_dim: int):
        super().__init__()

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.robot_encoder = MLPLayer(robot_obs_dim, 512, nn.Identity())
        self.motion_encoder = MLPLayer(motion_obs_dim, 512, nn.Identity())
        self.block_1 = AdaINBlock(512, 512)
        self.block_2 = AdaINBlock(512, 512)
        self.block_3 = AdaINBlock(512, 512)
        self.head = GaussianHead(512, action_dim)

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
        x = self.block_1(x_robot, x_motion)
        x = self.block_2(x, x_motion)
        x = self.block_3(x, x_motion)
        return self.head(x, action)


class AdaINResActor(nn.Module):
    def __init__(
        self,
        robot_obs_dim: int,
        motion_obs_dim: int,
        action_dim: int,
        num_blocks: int = 3,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

        self.robot_obs_normlizer = Normalizer((robot_obs_dim,))
        self.motion_obs_normlizer = Normalizer((motion_obs_dim,))
        self.num_blocks = num_blocks
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

        for block_idx in range(self.num_blocks):
            setattr(self, f"block_{block_idx + 1}", AdaINResBlock(512, 512))

        self.head = GaussianHead(512, action_dim)

    def _iter_blocks(self):
        for block_idx in range(self.num_blocks):
            yield getattr(self, f"block_{block_idx + 1}")

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
        x = x_robot
        for block in self._iter_blocks():
            x = block(x, x_motion)
        return self.head(x, action)


def build_actor(
    obs_dims: dict[str, int],
    actor_type: ActorType | str,
    action_dim: int,
    actor_kwargs: dict[str, int] | None = None,
) -> torch.nn.Module:
    actor_kwargs = actor_kwargs or {}
    normalized = normalize_actor_type(str(actor_type))
    if normalized == ActorType.VANILA:
        return VanilaActor(obs_dims["policy"], action_dim)
    if normalized == ActorType.RECURRENT:
        return RecurrentActor(
            obs_dims["policy"],
            action_dim,
            hidden_size=int(actor_kwargs.get("hidden_size", RecurrentActor.DEFAULT_HIDDEN_SIZE)),
            num_layers=int(actor_kwargs.get("num_layers", RecurrentActor.DEFAULT_NUM_LAYERS)),
        )
    if normalized == ActorType.SPLIT_ENCODER:
        return SplitEncoderActor(obs_dims["robot"], obs_dims["motion"], action_dim)
    if normalized == ActorType.ADAIN:
        return AdaINActor(obs_dims["robot"], obs_dims["motion"], action_dim)
    if normalized == ActorType.ADAIN_RES:
        return AdaINResActor(
            obs_dims["robot"],
            obs_dims["motion"],
            action_dim,
            num_blocks=int(actor_kwargs.get("num_blocks", 3)),
        )
    raise ValueError(f"Unsupported actor type '{actor_type}'.")


def get_actor_kwargs(
    actor: torch.nn.Module,
    actor_type: ActorType | str,
) -> dict[str, int]:
    normalized = normalize_actor_type(str(actor_type))
    if normalized == ActorType.ADAIN_RES:
        return {"num_blocks": int(actor.num_blocks)}
    if normalized == ActorType.RECURRENT:
        return {
            "hidden_size": int(actor.hidden_size),
            "num_layers": int(actor.num_layers),
        }
    return {}


def get_actor_observation(
    obs: dict[str, torch.Tensor],
    actor_type: ActorType | str,
) -> dict[str, torch.Tensor]:
    if is_concat_actor(actor_type):
        return {"obs": torch.cat((obs["motion"], obs["robot"]), dim=-1)}
    return {
        "motion_obs": obs["motion"],
        "robot_obs": obs["robot"],
    }


def get_policy_storage_specs(
    obs_dims: dict[str, int],
    actor_type: ActorType | str,
) -> dict[str, tuple[int, ...]]:
    if is_concat_actor(actor_type):
        return {"policy_observations": (obs_dims["policy"],)}
    return {
        "motion_observations": (obs_dims["motion"],),
        "robot_observations": (obs_dims["robot"],),
    }


def get_policy_records(
    actor_obs: dict[str, torch.Tensor],
    actor_type: ActorType | str,
) -> dict[str, torch.Tensor]:
    if is_concat_actor(actor_type):
        return {"policy_observations": actor_obs["obs"]}
    return {
        "motion_observations": actor_obs["motion_obs"],
        "robot_observations": actor_obs["robot_obs"],
    }


def get_policy_batch(
    batch: dict[str, torch.Tensor],
    actor_type: ActorType | str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if is_concat_actor(actor_type):
        return {"obs": batch["policy_observations"].to(device)}
    return {
        "motion_obs": batch["motion_observations"].to(device),
        "robot_obs": batch["robot_observations"].to(device),
    }
