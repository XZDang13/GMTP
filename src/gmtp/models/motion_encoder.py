from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn

from gmtp.integrations.ref2act.compat import _import_module
from gmtp.integrations.ref2act.observation_history import (
    build_gmtp_policy_observation_spec,
    build_motion_policy_window_lengths,
)
from gmtp.models.layers import MLPLayer, NormPosition
from gmtp.models.pooling import LearnedQueryAttentionPool
from gmtp.motion_mae import build_frozen_motion_mae_encoder

_OBSERVATION_SPEC = _import_module("ref2act.common.observation_spec")
DEFAULT_OBSERVATION_TERM_REGISTRY = _OBSERVATION_SPEC.DEFAULT_OBSERVATION_TERM_REGISTRY
ObservationLayout = _OBSERVATION_SPEC.ObservationLayout

MOTION_ENCODER_HIDDEN_DIM = 128
MOTION_ENCODER_OUTPUT_DIM = 512
MOTION_ENCODER_TRANSFORMER_HEADS = 4
MOTION_ENCODER_TRANSFORMER_LAYERS = 1
MOTION_ENCODER_FEEDFORWARD_DIM = 256


class MotionEncoderType(StrEnum):
    MLP = "mlp"
    TRANSFORMER = "transformer"
    MAE = "mae"


@dataclass(frozen=True)
class MotionWindowTermLayout:
    term_id: str
    term_dim: int
    flat_slice: slice
    step_slice: slice


@dataclass(frozen=True)
class MotionWindowLayout:
    window_length: int
    motion_obs_dim: int
    motion_step_dim: int
    terms: tuple[MotionWindowTermLayout, ...]


@lru_cache(maxsize=None)
def build_motion_window_layout(action_dim: int, motion_window_length: int) -> MotionWindowLayout:
    if motion_window_length < 1:
        raise ValueError(f"motion_window_length must be positive, got {motion_window_length}.")

    spec = build_gmtp_policy_observation_spec(
        add_noise=False,
        window_lengths=build_motion_policy_window_lengths(motion_window_length),
    )
    layout = ObservationLayout(joint_dim=action_dim, action_dim=action_dim, key_body_count=0)
    motion_group = next(group for group in spec.enabled_groups() if group.name == "motion")

    flat_offset = 0
    step_offset = 0
    motion_obs_dim = 0
    term_layouts: list[MotionWindowTermLayout] = []
    for term_spec in motion_group.terms:
        if not term_spec.enabled:
            continue
        if not term_spec.flatten and term_spec.window_length > 1:
            raise ValueError(
                "Motion history encoding only supports flattened motion observation history, "
                f"got term '{term_spec.id}' with flatten={term_spec.flatten} "
                f"and window_length={term_spec.window_length}."
            )
        if term_spec.window_length != motion_window_length:
            raise ValueError(
                "Motion history encoding requires a uniform motion window length, "
                f"got term '{term_spec.id}' with window_length={term_spec.window_length} "
                f"and expected {motion_window_length}."
            )

        term_dim = int(DEFAULT_OBSERVATION_TERM_REGISTRY[term_spec.type].dimension(layout, term_spec))
        flattened_dim = term_dim * term_spec.window_length if term_spec.flatten else term_dim
        term_layouts.append(
            MotionWindowTermLayout(
                term_id=term_spec.id,
                term_dim=term_dim,
                flat_slice=slice(flat_offset, flat_offset + flattened_dim),
                step_slice=slice(step_offset, step_offset + term_dim),
            )
        )
        flat_offset += flattened_dim
        step_offset += term_dim
        motion_obs_dim += flattened_dim

    return MotionWindowLayout(
        window_length=motion_window_length,
        motion_obs_dim=motion_obs_dim,
        motion_step_dim=step_offset,
        terms=tuple(term_layouts),
    )


def reshape_motion_history(motion_obs: torch.Tensor, layout: MotionWindowLayout) -> torch.Tensor:
    tensor = torch.as_tensor(motion_obs)
    if tensor.ndim not in {1, 2}:
        raise ValueError(f"Expected flat motion observation rank 1 or 2, got shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] != layout.motion_obs_dim:
        raise ValueError(
            f"Expected flat motion observation dim {layout.motion_obs_dim}, got {tensor.shape[-1]}."
        )

    batched = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    frames = []
    for term_layout in layout.terms:
        term_flat = batched[:, term_layout.flat_slice]
        frames.append(term_flat.reshape(batched.shape[0], layout.window_length, term_layout.term_dim))
    structured = torch.cat(frames, dim=-1)
    return structured.squeeze(0) if tensor.ndim == 1 else structured


def flatten_motion_history(motion_obs: torch.Tensor, layout: MotionWindowLayout) -> torch.Tensor:
    tensor = torch.as_tensor(motion_obs)
    expected_shape = (layout.window_length, layout.motion_step_dim)
    if tensor.ndim == 2:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Expected structured motion observation shape {expected_shape}, got {tuple(tensor.shape)}.")
        batched = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tuple(tensor.shape[-2:]) != expected_shape:
            raise ValueError(
                f"Expected structured motion observation trailing shape {expected_shape}, got {tuple(tensor.shape[-2:])}."
            )
        batched = tensor
    else:
        raise ValueError(f"Expected structured motion observation rank 2 or 3, got shape {tuple(tensor.shape)}.")

    flattened_terms = []
    for term_layout in layout.terms:
        flattened_terms.append(batched[:, :, term_layout.step_slice].reshape(batched.shape[0], -1))
    flat = torch.cat(flattened_terms, dim=-1)
    return flat.squeeze(0) if tensor.ndim == 2 else flat


def normalize_motion_encoder_type(motion_encoder_type: str | MotionEncoderType | None) -> MotionEncoderType:
    if motion_encoder_type is None:
        return MotionEncoderType.TRANSFORMER

    normalized = str(motion_encoder_type).lower().replace("-", "_")
    alias_map = {
        "mlp": MotionEncoderType.MLP,
        "transformer": MotionEncoderType.TRANSFORMER,
        "mae": MotionEncoderType.MAE,
        "motion_mae": MotionEncoderType.MAE,
        "pretrained_mae": MotionEncoderType.MAE,
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported motion encoder type '{motion_encoder_type}'. "
            "Expected one of: 'mlp', 'transformer', 'mae'."
        ) from exc


class MotionTransformerTokenEncoder(nn.Module):
    def __init__(self, *, motion_step_dim: int, motion_window_length: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MOTION_ENCODER_HIDDEN_DIM,
            nhead=MOTION_ENCODER_TRANSFORMER_HEADS,
            dim_feedforward=MOTION_ENCODER_FEEDFORWARD_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.input_proj = nn.Linear(motion_step_dim, MOTION_ENCODER_HIDDEN_DIM)
        self.position_embedding = nn.Parameter(torch.zeros(1, motion_window_length, MOTION_ENCODER_HIDDEN_DIM))
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=MOTION_ENCODER_TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(MOTION_ENCODER_HIDDEN_DIM),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, motion_obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(motion_obs)
        x = x + self.position_embedding
        return self.transformer(x)


class MotionWindowEncoder(nn.Module):
    def __init__(
        self,
        *,
        motion_step_dim: int,
        motion_window_length: int,
        motion_encoder_type: MotionEncoderType,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.motion_encoder_type = motion_encoder_type
        self._encoder_is_registered = motion_encoder_type != MotionEncoderType.MAE

        if motion_encoder_type == MotionEncoderType.MAE:
            if checkpoint_path is None:
                raise ValueError(
                    "motion_mae_encoder_checkpoint is required when motion_encoder_type='mae'."
                )
            frozen_encoder = build_frozen_motion_mae_encoder(checkpoint_path, device=device)
            schema = frozen_encoder.schema
            if int(frozen_encoder.encoder.past_frames) != int(motion_window_length):
                raise ValueError(
                    "Motion MAE encoder checkpoint is incompatible with motion_window_length: "
                    f"checkpoint past_frames={frozen_encoder.encoder.past_frames}, "
                    f"requested motion_window_length={motion_window_length}."
                )
            if int(schema.d_ref) != int(motion_step_dim):
                raise ValueError(
                    "Motion MAE encoder checkpoint is incompatible with policy motion dim: "
                    f"checkpoint d_ref={schema.d_ref}, policy motion step dim={motion_step_dim}."
                )
            if tuple(schema.reference_feature_names) != tuple(schema.policy_feature_names):
                raise ValueError(
                    "Motion MAE encoder checkpoint is incompatible with actor-integrated MAE mode: "
                    "reference_feature_names must match policy_feature_names."
                )

            object.__setattr__(self, "_encoder", frozen_encoder)
            token_dim = frozen_encoder.token_dim
            num_heads = frozen_encoder.num_heads
        else:
            self._encoder = MotionTransformerTokenEncoder(
                motion_step_dim=motion_step_dim,
                motion_window_length=motion_window_length,
            )
            token_dim = MOTION_ENCODER_HIDDEN_DIM
            num_heads = MOTION_ENCODER_TRANSFORMER_HEADS

        self.pooling = LearnedQueryAttentionPool(
            token_dim,
            num_heads,
        )
        self.output_proj = nn.Linear(token_dim, MOTION_ENCODER_OUTPUT_DIM)

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    def _apply(self, fn):
        super()._apply(fn)
        if not self._encoder_is_registered:
            object.__setattr__(self, "_encoder", self._encoder._apply(fn))
        return self

    def forward(self, motion_obs: torch.Tensor) -> torch.Tensor:
        if self.motion_encoder_type == MotionEncoderType.MAE:
            with torch.no_grad():
                tokens = self.encoder(motion_obs)
            tokens = tokens.detach()
        else:
            tokens = self.encoder(motion_obs)
        return self.output_proj(self.pooling(tokens))


class MotionHistoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        motion_obs_dim: int,
        action_dim: int,
        motion_window_length: int,
        motion_encoder_type: str | MotionEncoderType = MotionEncoderType.TRANSFORMER,
        motion_mae_encoder_checkpoint: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        self.motion_window_length = int(motion_window_length)
        self.motion_window_layout = build_motion_window_layout(action_dim, self.motion_window_length)
        if motion_obs_dim != self.motion_window_layout.motion_obs_dim:
            raise ValueError(
                f"Expected motion_obs_dim={self.motion_window_layout.motion_obs_dim} for action_dim={action_dim} "
                f"and motion_window_length={self.motion_window_length}, got {motion_obs_dim}."
            )

        self.motion_obs_dim = int(motion_obs_dim)
        self.motion_step_dim = int(self.motion_window_layout.motion_step_dim)
        self.is_windowed = self.motion_window_length > 1
        requested_encoder_type = normalize_motion_encoder_type(motion_encoder_type)

        if not self.is_windowed:
            self.motion_encoder_type = MotionEncoderType.MLP
            self.single_frame_encoder = nn.Sequential(
                MLPLayer(motion_obs_dim, MOTION_ENCODER_OUTPUT_DIM, nn.SiLU(), NormPosition.POST),
                MLPLayer(MOTION_ENCODER_OUTPUT_DIM, MOTION_ENCODER_OUTPUT_DIM, nn.SiLU(), NormPosition.POST),
                MLPLayer(MOTION_ENCODER_OUTPUT_DIM, MOTION_ENCODER_OUTPUT_DIM, nn.Identity()),
            )
            return

        if requested_encoder_type == MotionEncoderType.MLP:
            raise ValueError("Windowed motion observations require motion_encoder_type to be 'transformer' or 'mae'.")

        self.motion_encoder_type = requested_encoder_type
        self.window_encoder = MotionWindowEncoder(
            motion_step_dim=self.motion_step_dim,
            motion_window_length=self.motion_window_length,
            motion_encoder_type=self.motion_encoder_type,
            checkpoint_path=motion_mae_encoder_checkpoint,
            device=device,
        )

    def forward(self, motion_obs: torch.Tensor) -> torch.Tensor:
        if not self.is_windowed:
            if motion_obs.ndim != 2:
                raise ValueError(
                    f"Expected single-frame motion observation rank 2, got shape {tuple(motion_obs.shape)}."
                )
            if motion_obs.shape[-1] != self.motion_obs_dim:
                raise ValueError(
                    f"Expected single-frame motion observation dim {self.motion_obs_dim}, got {motion_obs.shape[-1]}."
                )
            return self.single_frame_encoder(motion_obs)

        if motion_obs.ndim != 3:
            raise ValueError(
                f"Expected windowed motion observation rank 3, got shape {tuple(motion_obs.shape)}."
            )
        expected_shape = (self.motion_window_length, self.motion_step_dim)
        if tuple(motion_obs.shape[-2:]) != expected_shape:
            raise ValueError(
                f"Expected windowed motion observation trailing shape {expected_shape}, "
                f"got {tuple(motion_obs.shape[-2:])}."
            )
        return self.window_encoder(motion_obs)
