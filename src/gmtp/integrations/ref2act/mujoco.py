from __future__ import annotations

from typing import Any

import torch

from .compat import load_mujoco_symbols


DEFAULT_ROOT_NAME = "torso_link"
DEFAULT_ANCHOR_BODY_NAME = "torso_link"


def normalize_action_mode(action_mode: object | None) -> str:
    normalized = str(action_mode or "absolute").split(".")[-1].lower().replace("-", "_")
    alias_map = {
        "absolute": "absolute",
        "median": "median",
        "offset": "offset",
        "residual": "residual",
        "current_residual": "current_residual",
        "currentresidual": "current_residual",
    }
    try:
        return alias_map[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported action mode '{action_mode}'.") from exc


def infer_legacy_action_mode(
    action_offset: torch.Tensor,
    action_scale: torch.Tensor,
    joint_pos_limits: torch.Tensor,
) -> str:
    lower_limits = joint_pos_limits[:, 0]
    upper_limits = joint_pos_limits[:, 1]
    median_offset = 0.5 * (upper_limits + lower_limits)
    median_scale = 0.5 * (upper_limits - lower_limits)

    if torch.allclose(action_offset, median_offset, atol=1e-5, rtol=1e-4) and torch.allclose(
        action_scale,
        median_scale,
        atol=1e-5,
        rtol=1e-4,
    ):
        return "median"

    if torch.allclose(action_offset, torch.zeros_like(action_offset), atol=1e-6, rtol=0.0):
        return "residual"

    return "offset"


def resolve_action_mode(
    checkpoint_env: dict[str, Any],
    action_mode: object | None,
    action_offset: torch.Tensor,
    action_scale: torch.Tensor,
    joint_pos_limits: torch.Tensor,
) -> tuple[str, str]:
    if action_mode is not None:
        return normalize_action_mode(action_mode), "argument"

    for key in ("action_mode", "action_mod"):
        checkpoint_action_mode = checkpoint_env.get(key)
        if checkpoint_action_mode is not None:
            return normalize_action_mode(checkpoint_action_mode), f"checkpoint:{key}"

    return infer_legacy_action_mode(action_offset, action_scale, joint_pos_limits), "inferred"


def resolve_name_override(
    explicit_value: str | None,
    checkpoint_env: dict[str, Any],
    checkpoint_keys: tuple[str, ...],
    default_value: str,
) -> str:
    if explicit_value is not None:
        return explicit_value

    for checkpoint_key in checkpoint_keys:
        checkpoint_value = checkpoint_env.get(checkpoint_key)
        if checkpoint_value:
            return str(checkpoint_value)

    return default_value


def get_mujoco_symbols():
    return load_mujoco_symbols()
