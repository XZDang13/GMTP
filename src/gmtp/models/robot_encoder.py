from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import torch
import torch.nn as nn

from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_policy_observation_spec,
    build_robot_policy_window_lengths,
)
from gmtp.models.layers import MLPLayer, NormPosition
from gmtp.models.pooling import (
    EncoderPoolingType,
    LastTokenPool,
    LearnedQueryAttentionPool,
    normalize_encoder_pooling_type,
)

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
DEFAULT_OBSERVATION_TERM_REGISTRY = _OBSERVATION_SPEC.DEFAULT_OBSERVATION_TERM_REGISTRY
ObservationLayout = _OBSERVATION_SPEC.ObservationLayout

ROBOT_ENCODER_HIDDEN_DIM = 128
ROBOT_ENCODER_OUTPUT_DIM = 512
ROBOT_ENCODER_TRANSFORMER_HEADS = 8
ROBOT_ENCODER_TRANSFORMER_LAYERS = 1
ROBOT_ENCODER_FEEDFORWARD_DIM = 256


class RobotEncoderType(StrEnum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


@dataclass(frozen=True)
class RobotWindowTermLayout:
    term_id: str
    term_dim: int
    flat_slice: slice
    step_slice: slice


@dataclass(frozen=True)
class RobotWindowLayout:
    window_length: int
    robot_obs_dim: int
    robot_step_dim: int
    terms: tuple[RobotWindowTermLayout, ...]


@lru_cache(maxsize=None)
def build_robot_window_layout(action_dim: int, robot_window_length: int) -> RobotWindowLayout:
    if robot_window_length < 1:
        raise ValueError(f"robot_window_length must be positive, got {robot_window_length}.")

    spec = build_gmtp_policy_observation_spec(
        add_noise=False,
        window_lengths=build_robot_policy_window_lengths(robot_window_length),
    )
    layout = ObservationLayout(joint_dim=action_dim, action_dim=action_dim, key_body_count=0)
    robot_group = next(group for group in spec.enabled_groups() if group.name == "robot")

    flat_offset = 0
    step_offset = 0
    robot_obs_dim = 0
    term_layouts: list[RobotWindowTermLayout] = []
    for term_spec in robot_group.terms:
        if not term_spec.enabled:
            continue
        if not term_spec.flatten and term_spec.window_length > 1:
            raise ValueError(
                "Robot history encoding only supports flattened robot observation history, "
                f"got term '{term_spec.id}' with flatten={term_spec.flatten} "
                f"and window_length={term_spec.window_length}."
            )
        if term_spec.window_length != robot_window_length:
            raise ValueError(
                "Robot history encoding requires a uniform robot window length, "
                f"got term '{term_spec.id}' with window_length={term_spec.window_length} "
                f"and expected {robot_window_length}."
            )

        term_dim = int(DEFAULT_OBSERVATION_TERM_REGISTRY[term_spec.type].dimension(layout, term_spec))
        flattened_dim = term_dim * term_spec.window_length if term_spec.flatten else term_dim
        term_layouts.append(
            RobotWindowTermLayout(
                term_id=term_spec.id,
                term_dim=term_dim,
                flat_slice=slice(flat_offset, flat_offset + flattened_dim),
                step_slice=slice(step_offset, step_offset + term_dim),
            )
        )
        flat_offset += flattened_dim
        step_offset += term_dim
        robot_obs_dim += flattened_dim

    return RobotWindowLayout(
        window_length=robot_window_length,
        robot_obs_dim=robot_obs_dim,
        robot_step_dim=step_offset,
        terms=tuple(term_layouts),
    )


def reshape_robot_history(robot_obs: torch.Tensor, layout: RobotWindowLayout) -> torch.Tensor:
    tensor = torch.as_tensor(robot_obs)
    if tensor.ndim not in {1, 2}:
        raise ValueError(f"Expected flat robot observation rank 1 or 2, got shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] != layout.robot_obs_dim:
        raise ValueError(
            f"Expected flat robot observation dim {layout.robot_obs_dim}, got {tensor.shape[-1]}."
        )

    batched = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    frames = []
    for term_layout in layout.terms:
        term_flat = batched[:, term_layout.flat_slice]
        frames.append(term_flat.reshape(batched.shape[0], layout.window_length, term_layout.term_dim))
    structured = torch.cat(frames, dim=-1)
    return structured.squeeze(0) if tensor.ndim == 1 else structured


def flatten_robot_history(robot_obs: torch.Tensor, layout: RobotWindowLayout) -> torch.Tensor:
    tensor = torch.as_tensor(robot_obs)
    expected_shape = (layout.window_length, layout.robot_step_dim)
    if tensor.ndim == 2:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Expected structured robot observation shape {expected_shape}, got {tuple(tensor.shape)}.")
        batched = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tuple(tensor.shape[-2:]) != expected_shape:
            raise ValueError(
                f"Expected structured robot observation trailing shape {expected_shape}, got {tuple(tensor.shape[-2:])}."
            )
        batched = tensor
    else:
        raise ValueError(f"Expected structured robot observation rank 2 or 3, got shape {tuple(tensor.shape)}.")

    flattened_terms = []
    for term_layout in layout.terms:
        flattened_terms.append(batched[:, :, term_layout.step_slice].reshape(batched.shape[0], -1))
    flat = torch.cat(flattened_terms, dim=-1)
    return flat.squeeze(0) if tensor.ndim == 2 else flat


def normalize_robot_encoder_type(robot_encoder_type: str | RobotEncoderType | None) -> RobotEncoderType:
    if robot_encoder_type is None:
        return RobotEncoderType.TRANSFORMER

    normalized = str(robot_encoder_type).lower().replace("-", "_")
    alias_map = {
        "mlp": RobotEncoderType.MLP,
        "transformer": RobotEncoderType.TRANSFORMER,
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported robot encoder type '{robot_encoder_type}'. "
            "Windowed robot observations support only 'transformer'."
        ) from exc


class RobotTransformerTokenEncoder(nn.Module):
    def __init__(self, *, robot_step_dim: int, robot_window_length: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ROBOT_ENCODER_HIDDEN_DIM,
            nhead=ROBOT_ENCODER_TRANSFORMER_HEADS,
            dim_feedforward=ROBOT_ENCODER_FEEDFORWARD_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.input_proj = nn.Linear(robot_step_dim, ROBOT_ENCODER_HIDDEN_DIM)
        self.position_embedding = nn.Parameter(torch.zeros(1, robot_window_length, ROBOT_ENCODER_HIDDEN_DIM))
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ROBOT_ENCODER_TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(ROBOT_ENCODER_HIDDEN_DIM),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(robot_obs)
        x = x + self.position_embedding
        return self.transformer(x)


class RobotWindowEncoder(nn.Module):
    def __init__(
        self,
        *,
        robot_step_dim: int,
        robot_window_length: int,
        encoder_pooling_type: str | EncoderPoolingType = EncoderPoolingType.LEARNED,
    ) -> None:
        super().__init__()
        self.encoder_pooling_type = normalize_encoder_pooling_type(encoder_pooling_type)
        self._encoder = RobotTransformerTokenEncoder(
            robot_step_dim=robot_step_dim,
            robot_window_length=robot_window_length,
        )
        if self.encoder_pooling_type is EncoderPoolingType.LEARNED:
            self.pooling = LearnedQueryAttentionPool(
                ROBOT_ENCODER_HIDDEN_DIM,
                ROBOT_ENCODER_TRANSFORMER_HEADS,
            )
        else:
            self.pooling = LastTokenPool(ROBOT_ENCODER_HIDDEN_DIM)
        self.output_proj = nn.Linear(ROBOT_ENCODER_HIDDEN_DIM, ROBOT_ENCODER_OUTPUT_DIM)

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(robot_obs)
        return self.output_proj(self.pooling(tokens))


class RobotHistoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        robot_obs_dim: int,
        action_dim: int,
        robot_window_length: int,
        robot_encoder_type: str | RobotEncoderType = RobotEncoderType.TRANSFORMER,
        encoder_pooling_type: str | EncoderPoolingType = EncoderPoolingType.LEARNED,
    ) -> None:
        super().__init__()

        self.robot_window_length = int(robot_window_length)
        self.robot_window_layout = build_robot_window_layout(action_dim, self.robot_window_length)
        if robot_obs_dim != self.robot_window_layout.robot_obs_dim:
            raise ValueError(
                f"Expected robot_obs_dim={self.robot_window_layout.robot_obs_dim} for action_dim={action_dim} "
                f"and robot_window_length={self.robot_window_length}, got {robot_obs_dim}."
            )

        self.robot_obs_dim = int(robot_obs_dim)
        self.robot_step_dim = int(self.robot_window_layout.robot_step_dim)
        self.is_windowed = self.robot_window_length > 1
        requested_encoder_type = normalize_robot_encoder_type(robot_encoder_type)
        self.encoder_pooling_type = normalize_encoder_pooling_type(encoder_pooling_type)

        if not self.is_windowed:
            self.robot_encoder_type = RobotEncoderType.MLP
            self.single_frame_encoder = nn.Sequential(
                MLPLayer(robot_obs_dim, ROBOT_ENCODER_OUTPUT_DIM, nn.SiLU(), NormPosition.POST),
                MLPLayer(ROBOT_ENCODER_OUTPUT_DIM, ROBOT_ENCODER_OUTPUT_DIM, nn.Identity()),
            )
            return

        if requested_encoder_type != RobotEncoderType.TRANSFORMER:
            raise ValueError("Windowed robot observations require robot_encoder_type='transformer'.")

        self.robot_encoder_type = requested_encoder_type
        self.window_encoder = RobotWindowEncoder(
            robot_step_dim=self.robot_step_dim,
            robot_window_length=self.robot_window_length,
            encoder_pooling_type=self.encoder_pooling_type,
        )

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        if not self.is_windowed:
            if robot_obs.ndim != 2:
                raise ValueError(
                    f"Expected single-frame robot observation rank 2, got shape {tuple(robot_obs.shape)}."
                )
            if robot_obs.shape[-1] != self.robot_obs_dim:
                raise ValueError(
                    f"Expected single-frame robot observation dim {self.robot_obs_dim}, got {robot_obs.shape[-1]}."
                )
            return self.single_frame_encoder(robot_obs)

        if robot_obs.ndim != 3:
            raise ValueError(
                f"Expected windowed robot observation rank 3, got shape {tuple(robot_obs.shape)}."
            )
        expected_shape = (self.robot_window_length, self.robot_step_dim)
        if tuple(robot_obs.shape[-2:]) != expected_shape:
            raise ValueError(
                f"Expected windowed robot observation trailing shape {expected_shape}, got {tuple(robot_obs.shape[-2:])}."
            )
        return self.window_encoder(robot_obs)
