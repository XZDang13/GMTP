from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
from RLAlg.nn.layers import CriticHead
from RLAlg.nn.steps import ValueStep
from RLAlg.normalizer import Normalizer

from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import build_gmtp_observation_spec

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


@lru_cache(maxsize=None)
def build_critic_privilege_layout(action_dim: int) -> CriticPrivilegeLayout:
    action_dim = int(action_dim)
    if action_dim < 1:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")

    spec = build_gmtp_observation_spec(add_noise=False)
    layout = ObservationLayout(joint_dim=action_dim, action_dim=action_dim, key_body_count=0)
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
    def __init__(self, obs_dim: int, action_dim: int | None = None):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.normlizer = Normalizer((self.obs_dim,))
        self.critic_type = "flat"
        self.privilege_layout: CriticPrivilegeLayout | None = None

        if action_dim is not None:
            privilege_layout = build_critic_privilege_layout(action_dim)
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

    def _encode_structured(self, obs: torch.Tensor) -> torch.Tensor:
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

    def forward(self, obs: torch.Tensor, update_normlizer: bool = False) -> ValueStep:
        obs = self.normlizer(obs, update=update_normlizer)
        if self.critic_type == "structured":
            x = self._encode_structured(obs)
        else:
            x = self.encoder(obs)
        return self.head(x)
