from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from RLAlg.nn.layers import CriticHead
from RLAlg.nn.steps import ValueStep
from RLAlg.normalizer import Normalizer

from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_observation_spec,
    normalize_observation_window_lengths,
)

from .layers import MLPLayer, NormPosition

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
DEFAULT_OBSERVATION_TERM_REGISTRY = _OBSERVATION_SPEC.DEFAULT_OBSERVATION_TERM_REGISTRY
ObservationLayout = _OBSERVATION_SPEC.ObservationLayout

CRITIC_HIDDEN_DIM = 512
CRITIC_GROUP_HIDDEN_DIM = 256
CRITIC_PRIVILEGE_TERM_GROUPS = (
    ("target", ("priv_target_projected_gravity", "priv_target_joint_pos", "priv_target_joint_vel")),
    (
        "geometry",
        (
            "relative_anchor_pos",
            "relative_anchor_tangent_and_normal",
            "relative_key_pos",
            "relative_key_tangent_and_normal",
        ),
    ),
    (
        "robot",
        (
            "priv_projected_gravity",
            "anchor_lin_vel",
            "priv_anchor_ang_vel_b",
            "priv_joint_pos",
            "priv_joint_vel",
            "priv_previous_action",
        ),
    ),
)


@dataclass(frozen=True)
class CriticPrivilegeTermLayout:
    term_id: str
    term_dim: int
    flat_dim: int
    flat_slice: slice


@dataclass(frozen=True)
class CriticPrivilegeGroupLayout:
    name: str
    term_ids: tuple[str, ...]
    flat_slices: tuple[slice, ...]
    obs_dim: int


@dataclass(frozen=True)
class CriticPrivilegeLayout:
    obs_dim: int
    terms: tuple[CriticPrivilegeTermLayout, ...]
    groups: tuple[CriticPrivilegeGroupLayout, ...]

    def group(self, name: str) -> CriticPrivilegeGroupLayout:
        for group_layout in self.groups:
            if group_layout.name == name:
                return group_layout
        raise KeyError(f"Unknown critic privilege group '{name}'.")


def _window_lengths_cache_key(
    observation_window_lengths: Mapping[str, int] | None,
) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(normalize_observation_window_lengths(observation_window_lengths).items()))


@lru_cache(maxsize=None)
def _build_critic_privilege_layout_cached(
    action_dim: int,
    key_body_count: int,
    window_lengths_key: tuple[tuple[str, int], ...],
) -> CriticPrivilegeLayout:
    action_dim = int(action_dim)
    if action_dim < 1:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    key_body_count = int(key_body_count)
    if key_body_count < 0:
        raise ValueError(f"key_body_count must be non-negative, got {key_body_count}.")

    spec = build_gmtp_observation_spec(add_noise=False, window_lengths=dict(window_lengths_key) or None)
    layout = ObservationLayout(joint_dim=action_dim, action_dim=action_dim, key_body_count=key_body_count)
    privilege_group = next(group for group in spec.enabled_groups() if group.name == "privilege")

    offset = 0
    term_layouts = []
    for term_spec in privilege_group.terms:
        if not term_spec.enabled:
            continue
        term_dim = int(DEFAULT_OBSERVATION_TERM_REGISTRY[term_spec.type].dimension(layout, term_spec))
        flat_dim = term_dim * term_spec.window_length if term_spec.flatten else term_dim
        term_layouts.append(
            CriticPrivilegeTermLayout(
                term_id=term_spec.id,
                term_dim=term_dim,
                flat_dim=flat_dim,
                flat_slice=slice(offset, offset + flat_dim),
            )
        )
        offset += flat_dim

    terms_by_id = {term_layout.term_id: term_layout for term_layout in term_layouts}
    group_layouts = []
    for group_name, term_ids in CRITIC_PRIVILEGE_TERM_GROUPS:
        missing_term_ids = [term_id for term_id in term_ids if term_id not in terms_by_id]
        if missing_term_ids:
            raise ValueError(f"Missing privilege terms for critic group '{group_name}': {missing_term_ids}.")
        flat_slices = tuple(terms_by_id[term_id].flat_slice for term_id in term_ids)
        group_layouts.append(
            CriticPrivilegeGroupLayout(
                name=group_name,
                term_ids=term_ids,
                flat_slices=flat_slices,
                obs_dim=sum(item.stop - item.start for item in flat_slices),
            )
        )

    return CriticPrivilegeLayout(
        obs_dim=offset,
        terms=tuple(term_layouts),
        groups=tuple(group_layouts),
    )


def build_critic_privilege_layout(
    action_dim: int,
    *,
    key_body_count: int = 0,
    observation_window_lengths: Mapping[str, int] | None = None,
) -> CriticPrivilegeLayout:
    return _build_critic_privilege_layout_cached(
        int(action_dim),
        int(key_body_count),
        _window_lengths_cache_key(observation_window_lengths),
    )


def _build_flat_encoder(obs_dim: int) -> nn.Sequential:
    return nn.Sequential(
        MLPLayer(obs_dim, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
    )


def _build_group_encoder(obs_dim: int) -> nn.Sequential:
    return nn.Sequential(
        MLPLayer(obs_dim, CRITIC_GROUP_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        MLPLayer(CRITIC_GROUP_HIDDEN_DIM, CRITIC_GROUP_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
    )


def _build_fusion_encoder(group_count: int) -> nn.Sequential:
    return nn.Sequential(
        MLPLayer(group_count * CRITIC_GROUP_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
        MLPLayer(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM, nn.SiLU(), NormPosition.POST),
    )


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int | None = None,
        *,
        key_body_count: int = 0,
        observation_window_lengths: Mapping[str, int] | None = None,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.key_body_count = int(key_body_count)
        self.observation_window_lengths = normalize_observation_window_lengths(observation_window_lengths)
        self.normlizer = Normalizer((self.obs_dim,))
        self.critic_type = "flat"
        self.privilege_layout: CriticPrivilegeLayout | None = None

        if action_dim is not None:
            privilege_layout = build_critic_privilege_layout(
                action_dim,
                key_body_count=self.key_body_count,
                observation_window_lengths=self.observation_window_lengths,
            )
            if privilege_layout.obs_dim == self.obs_dim:
                self.critic_type = "structured"
                self.privilege_layout = privilege_layout

        if self.critic_type == "structured":
            assert self.privilege_layout is not None
            self.target_encoder = _build_group_encoder(self.privilege_layout.group("target").obs_dim)
            self.geometry_encoder = _build_group_encoder(self.privilege_layout.group("geometry").obs_dim)
            self.robot_encoder = _build_group_encoder(self.privilege_layout.group("robot").obs_dim)
            self.fusion_encoder = _build_fusion_encoder(len(self.privilege_layout.groups))
        else:
            self.encoder = _build_flat_encoder(self.obs_dim)
        self.head = CriticHead(CRITIC_HIDDEN_DIM)

    @property
    def is_structured(self) -> bool:
        return self.critic_type == "structured" and self.privilege_layout is not None

    def observation_storage_specs(self) -> dict[str, tuple[int, ...]]:
        if self.privilege_layout is None:
            return {"critic_observations": (self.obs_dim,)}
        return {
            f"critic_{group_layout.name}_observations": (group_layout.obs_dim,)
            for group_layout in self.privilege_layout.groups
        }

    def split_observation(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.privilege_layout is None:
            raise RuntimeError("Structured critic observation split requires a privilege layout.")

        obs = torch.as_tensor(obs)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected critic observation dim {self.obs_dim}, got {obs.shape[-1]}.")
        return {
            group_layout.name: torch.cat([obs[..., flat_slice] for flat_slice in group_layout.flat_slices], dim=-1)
            for group_layout in self.privilege_layout.groups
        }

    def flatten_observation(self, obs: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(obs, Mapping):
            tensor = torch.as_tensor(obs)
            if tensor.shape[-1] != self.obs_dim:
                raise ValueError(f"Expected critic observation dim {self.obs_dim}, got {tensor.shape[-1]}.")
            return tensor
        if self.privilege_layout is None:
            raise RuntimeError("Flat critic cannot consume structured observation mappings.")

        group_tensors = []
        for group_layout in self.privilege_layout.groups:
            if group_layout.name not in obs:
                raise KeyError(f"Structured critic observation is missing group '{group_layout.name}'.")
            group_obs = torch.as_tensor(obs[group_layout.name])
            if group_obs.shape[-1] != group_layout.obs_dim:
                raise ValueError(
                    f"Expected critic {group_layout.name} observation dim {group_layout.obs_dim}, "
                    f"got {group_obs.shape[-1]}."
                )
            group_tensors.append(group_obs)
        return torch.cat(group_tensors, dim=-1)

    def _encode_structured_flat(self, obs: torch.Tensor) -> torch.Tensor:
        if self.privilege_layout is None:
            raise RuntimeError("Structured critic encoder requires a privilege layout.")

        squeeze_batch = obs.ndim == 1
        batched_obs = obs.unsqueeze(0) if squeeze_batch else obs
        embeddings = []
        for group_layout in self.privilege_layout.groups:
            group_obs = torch.cat([batched_obs[..., flat_slice] for flat_slice in group_layout.flat_slices], dim=-1)
            group_encoder = getattr(self, f"{group_layout.name}_encoder")
            embeddings.append(group_encoder(group_obs))

        x = self.fusion_encoder(torch.cat(embeddings, dim=-1))
        return x.squeeze(0) if squeeze_batch else x

    def forward(
        self,
        obs: torch.Tensor | Mapping[str, torch.Tensor],
        update_normlizer: bool = False,
    ) -> ValueStep:
        flat_obs = self.flatten_observation(obs)
        flat_obs = self.normlizer(flat_obs, update=update_normlizer)
        if self.critic_type == "structured":
            x = self._encode_structured_flat(flat_obs)
        else:
            x = self.encoder(flat_obs)
        return self.head(x)


def get_critic_observation(
    critic: Critic,
    obs: Mapping[str, torch.Tensor],
) -> torch.Tensor | dict[str, torch.Tensor]:
    privilege_obs = obs["privilege"]
    if not critic.is_structured:
        return privilege_obs
    return critic.split_observation(privilege_obs)


def get_critic_records(
    critic_obs: torch.Tensor | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(critic_obs, Mapping):
        return {"critic_observations": torch.as_tensor(critic_obs)}
    return {f"critic_{name}_observations": torch.as_tensor(value) for name, value in critic_obs.items()}


def get_critic_batch(
    batch: Mapping[str, torch.Tensor],
    critic: Critic,
    device: torch.device | str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if critic.privilege_layout is None:
        return batch["critic_observations"].to(device)
    return {
        group_layout.name: batch[f"critic_{group_layout.name}_observations"].to(device)
        for group_layout in critic.privilege_layout.groups
    }


def get_critic_metadata(critic: Critic) -> dict[str, Any]:
    group_dims = {}
    if critic.privilege_layout is not None:
        group_dims = {group_layout.name: group_layout.obs_dim for group_layout in critic.privilege_layout.groups}
    return {
        "critic_type": critic.critic_type,
        "obs_dim": critic.obs_dim,
        "key_body_count": critic.key_body_count,
        "group_dims": group_dims,
    }
